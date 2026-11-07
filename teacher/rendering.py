"""Optional synchronized review cameras; never advance physics to render."""
import io
import json
import numpy as np

from teacher.transport import PROTOCOL,StateStore,handler_for

CAMERAS={'blue':'/World/so100_blue/gripper/Camera','red':'/World/so100_red/gripper/Camera'}


def pack_pair(images,observation,sequence):
    if set(images)!=set(CAMERAS): raise ValueError('Need both authored cameras')
    shapes={image.shape for image in images.values()}
    if len(shapes)!=1 or any(v.dtype!=np.uint8 or v.ndim!=3 or v.shape[-1]!=3 for v in images.values()):
        raise ValueError('Need matching uint8 RGB camera pairs')
    stream=io.BytesIO()
    np.savez(stream,**images,frame_id=int(sequence),simulation_time=observation['physics_time'],
             observation_json=np.array(json.dumps(observation)))
    return stream.getvalue()


class CameraCapture:
    def __init__(self,stage,control,resolution=(640,480)):
        import omni.replicator.core as rep
        from pxr import UsdGeom
        self.rep=rep;self.control=control;self.resolution=tuple(resolution);self.products=[];self.annotators={}
        rep.orchestrator.set_capture_on_play(False)
        for name,path in CAMERAS.items():
            prim=stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Camera): raise ValueError('Missing authored camera '+path)
            product=rep.create.render_product(path,self.resolution,force_new=True,name='teacher_review_'+name)
            self.products.append(product)
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product);self.annotators[name]=annotator
        before=self.physical_state()
        for _ in range(10): rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True)
        self.check_unchanged(before)
    def physical_state(self):
        driver=self.control.driver
        positions,rotations=driver.pieces.view.get_world_poses()
        values=[np.array([driver.world.current_time]),positions.copy(),rotations.copy(),
            driver.pieces.view.get_velocities().copy()]
        for robot in driver.robots.values():
            values.extend([robot.get_joint_positions().copy(),robot.get_joint_velocities().copy()])
        return values
    def check_unchanged(self,before):
        after=self.physical_state()
        errors=[float(np.max(np.abs(a-b))) for a,b in zip(after,before)]
        if max(errors)>1e-6: raise RuntimeError('Rendering changed physics state: '+str(errors))
        return max(errors)
    def capture(self,observation,sequence):
        from PIL import Image
        before=self.physical_state()
        self.rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True)
        images={};jpegs={}
        for name,annotator in self.annotators.items():
            raw=annotator.get_data()
            if isinstance(raw,dict): raw=raw.get('data')
            rgba=np.asarray(raw)
            if rgba.shape!=(self.resolution[1],self.resolution[0],4) or rgba.dtype!=np.uint8:
                raise RuntimeError('Missing paired review image for '+name)
            images[name]=np.ascontiguousarray(rgba[...,:3])
            stream=io.BytesIO();Image.fromarray(images[name]).save(stream,format='JPEG',quality=90)
            jpegs[name]=stream.getvalue()
        drift=self.check_unchanged(before)
        return pack_pair(images,observation,sequence),jpegs,{'physics_drift_max':drift,'resolution':list(self.resolution)}
    def close(self):
        for annotator,product in zip(self.annotators.values(),self.products): annotator.detach(product)
        for product in self.products: product.destroy()


class CameraStore(StateStore):
    def __init__(self,capture):
        super().__init__();self.capture=capture;self.pair=None;self.jpegs={};self.camera_stats={}
    def publish(self,observation,sequence,timing=None,reset_context=None):
        pair,jpegs,stats=self.capture.capture(observation,sequence)
        snapshot=dict(protocol=PROTOCOL,sequence=sequence,simulation_time=observation['physics_time'],
            observation=observation,timing_ms=timing or {},camera_render_products=2,camera_stats=stats)
        if reset_context is not None: snapshot['reset_context']=reset_context
        # A reader can never see a new state with an old pair or vice versa.
        with self.lock:
            self.snapshot=snapshot;self.pair=pair;self.jpegs=jpegs;self.camera_stats=stats
        return snapshot


def camera_handler_for(store):
    parent=handler_for(store)
    class Handler(parent):
        def do_GET(self):
            if self.path=='/state': return super().do_GET()
            path=self.path.split('?',1)[0]
            if path=='/':
                body=('<!doctype html><meta charset="utf-8"><title>Unified teacher review</title>'
                    '<style>body{background:#16181c;color:#eee;font:18px sans-serif}section{display:flex;flex-wrap:wrap}'
                    'img{width:640px;max-width:100%}</style><h1>Frozen state-teacher review</h1>'
                    '<p id="status">Waiting for control frames</p><section><div>Fixed red camera<br><img id="red"></div>'
                    '<div>Blue wrist camera<br><img id="blue"></div></section><script>let last=-1;'
                    'async function update(){try{let s=await(await fetch("/state")).json();if(s.sequence!==last){'
                    'last=s.sequence;for(let c of ["red","blue"])document.getElementById(c).src="/"+c+".jpg?s="+last;'
                    'document.getElementById("status").textContent="Control step "+s.observation.control_step+'
                    '" · cameras used for review; policy uses simulator state";}}catch(e){} }update();setInterval(update,500);'
                    '</script>').encode();mime='text/html'
            else:
                with store.lock:
                    if path=='/frames.npz': body=store.pair;mime='application/octet-stream'
                    elif path in ('/blue.jpg','/red.jpg'): body=store.jpegs.get(path[1:-4]);mime='image/jpeg'
                    else: self.send_error(404);return
            if body is None: self.send_error(503);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError): pass
    return Handler
