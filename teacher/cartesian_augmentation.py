"""Geometry-consistent corrective BC labels, never simulator/replay samples."""
import torch


def augment_cartesian(inputs,labels,aux,noise,midpoint,low,high,config):
    inputs=inputs.clone();labels=labels.clone();batch=len(inputs)
    q=inputs[:,:6]*180;command=q+inputs[:,23:29]*5
    held=(~aux['released_before']) & (inputs[:,17]*.03>.008)
    eligible=aux['free_arm_before'] | held
    noise=noise*eligible[:,None];noise[:,5]=0
    qnew=(q+noise).clamp(low,high);offset=qnew-q;cmdnew=(command+offset).clamp(low,high)
    cmd_pose=command.clone();cmd_pose[:,5]=q[:,5]
    cmd_new_pose=cmdnew.clone();cmd_new_pose[:,5]=qnew[:,5]
    jaw2=inputs[:,22]*10
    with torch.no_grad():
        points,rotations=midpoint.forward_pose(torch.cat([q,qnew,cmd_pose,cmd_new_pose]),jaw2.repeat(4))
    p,pnew,pcmd,pcmdnew=points.chunk(4);r,rnew,_,_=rotations.chunk(4)
    relative_rotation=r.transpose(1,2)@rnew
    def rotate(v): return (v[:,None,:]@relative_rotation)[:,0,:]
    centers=aux['all_centers_m'].clone();uprights=aux['all_upright'].clone()
    source=aux['source_index'].long();indices=torch.arange(batch)
    center=centers[indices,source];grasp=p+inputs[:,12:15]*.03
    center_new=torch.where(held[:,None],pnew+rotate(center-p),center)
    grasp_new=torch.where(held[:,None],pnew+rotate(grasp-p),grasp)
    up=torch.where(held[:,None],rotate(inputs[:,68:71]),inputs[:,68:71])
    inputs[:,:6]=qnew/180;inputs[:,23:29]=(cmdnew-qnew)/5
    inputs[:,12:15]=(grasp_new-pnew)/.03
    inputs[:,15:18]+=(center_new-center)/.03;inputs[:,18:21]-=(center_new-center)/.1
    # Material grasp targets can lie below a piece's COM (e.g. a knight).
    # Preserve the distinction between that signed axis and actual uprightness.
    axis_sign=torch.where(inputs[:,41]*inputs[:,70]<0,-1.,1.)
    true_up_z=up[:,2]*axis_sign
    inputs[:,41]=true_up_z;inputs[:,68:71]=up
    inputs[:,35:38]=torch.where(held[:,None],rotate(inputs[:,35:38]),inputs[:,35:38])
    inputs[:,38:41]=torch.where(held[:,None],rotate(inputs[:,38:41]),inputs[:,38:41])
    centers[indices,source]=center_new;uprights[indices,source]=true_up_z
    distance=torch.linalg.norm(centers-pnew[:,None,:],dim=2);distance[indices,source]=float('inf')
    nearest=distance.topk(4,largest=False).indices
    neighbours=centers.gather(1,nearest[:,:,None].expand(-1,-1,3))
    neighbours_up=uprights.gather(1,nearest)
    inputs[:,42:58]=torch.cat([(neighbours-pnew[:,None,:])/.05,neighbours_up[:,:,None]],dim=2).flatten(1)
    inputs[:,62:67]=(aux['home_joint_deg'][:,:5]-qnew[:,:5])/90
    inputs[:,71:74]-=(pnew-p)/.2
    inputs[:,74]-=offset[:,1:4].sum(1)/90;inputs[:,75]-=offset[:,4]/90
    # Restore the same intended next task-space pose from the perturbed
    # command pose. No reward/dynamics/next-state transition is invented.
    labels[:,:3]-=(pcmdnew-pcmd)/config['translation_step_m']
    labels[:,3]-=(cmdnew-command)[:,1:4].sum(1)/config['pitch_step_deg']
    labels[:,4]-=(cmdnew-command)[:,4]/config['roll_step_deg']
    labels[:,:5]=labels[:,:5].clamp(-1,1)
    return inputs,labels
