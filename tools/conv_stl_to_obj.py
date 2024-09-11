# usage: blender --background --python conv_stl_to_obj.py -- gym_drake_lca/assets/*.stl
import sys

import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1 :]  # get all args after "--"

for stl_in in argv:
    print(stl_in)
    obj_out = stl_in.replace("stl", "obj")
    print(obj_out)

    bpy.ops.wm.stl_import(filepath=stl_in, forward_axis="X", up_axis="Z")
    bpy.ops.wm.obj_export(filepath=obj_out, forward_axis="X", up_axis="Z")

    bpy.ops.wm.read_factory_settings(use_empty=True)
