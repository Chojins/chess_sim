"""Keep collision-geometry variants out of each other's physical replay."""
import copy
import json
from pathlib import Path


def physics_signature(settings):
    value=copy.deepcopy(settings)
    if 'magnets' in value: value['magnets'].pop('initial_settling',None)
    return value


def source_physics(manifest,sources):
    expected=physics_signature(manifest['physics_settings']) if 'physics_settings' in manifest else None
    for source in sources:
        initial=json.loads((Path(source['path'])/'initial.json').read_text())['telemetry']['observation']
        value=physics_signature(initial['physics_settings'])
        if expected is not None and value!=expected: raise ValueError('Physical source collision/dynamics contract differs')
        expected=value
    if expected is None: raise ValueError('No recorded physical source contract')
    return expected
