import xml.etree.ElementTree as ET
import xml.dom.minidom
import math
import os
import sys
from collections import namedtuple
import subprocess
from functools import partial, reduce

# Try importing bpy, handle its absence gracefully
try:
    import bpy
except ImportError:
    bpy = None
    print("bpy module not available. Running outside of Blender.")

# Paths configuration
blender_executable_path = '/Applications/Blender.app/Contents/MacOS/Blender'
blender_file_path = 'Alfredo_Armature_1.blend'
mujoco_file_path = 'output_mujoco.xml'
export_directory = './assets/'

# Named tuple for storing bone information
BoneInfo = namedtuple('BoneInfo', ['head', 'axis', 'length', 'parent', 'joint_type', 'mesh'])

def create_mujoco_xml():
    mujoco = ET.Element('mujoco', model='blender_export')
    ET.SubElement(mujoco, 'compiler', angle='radian', meshdir='assets', texturedir='assets', autolimits='true')
    default = ET.SubElement(mujoco, 'default')
    ET.SubElement(default, 'geom', type='mesh')
    asset = ET.SubElement(mujoco, 'asset')
    worldbody = ET.SubElement(mujoco, 'worldbody')
    return mujoco, worldbody, asset

def vector_to_string(vector):
    return " ".join(str(v) for v in vector)

def calculate_axis(bone):
    axis_vector = bone.tail_local - bone.head_local
    axis_vector.normalize()
    return axis_vector

def export_mesh_as_stl(mesh_name, export_path):
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    mesh = bpy.data.objects.get(mesh_name)
    if not mesh:
        return

    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.export_mesh.stl(filepath=export_path, use_selection=True, axis_forward='-Z', axis_up='Y')

def infer_joint_type(bone, armature):
    # Check for a direct specification of the joint type via custom properties
    mujoco_joint_type = bone.get("mujoco_joint_type")
    if mujoco_joint_type:
        return mujoco_joint_type, (0, 0, 0)  # Adjust the default axis vector as needed

    joint = armature.pose.bones[bone.name]

    rotation_limits = {
        'X': (joint.lock_ik_x, joint.ik_min_x, joint.ik_max_x),
        'Y': (joint.lock_ik_y, joint.ik_min_y, joint.ik_max_y),
        'Z': (joint.lock_ik_z, joint.ik_min_z, joint.ik_max_z),
    }

    # Determine the primary axis of movement based on IK locking
    primary_axis = None
    for axis, (locked, min_limit, max_limit) in rotation_limits.items():
        if not locked:  # If the axis is not locked, it's a candidate for being the primary axis
            if primary_axis is None or (min_limit, max_limit) != (0, 0):
                primary_axis = axis

    # Convert the primary axis to a 3-vector
    axis_vector = {
        'X': (1, 0, 0),
        'Y': (0, 1, 0),
        'Z': (0, 0, 1)
    }.get(primary_axis, (0, 0, 0))  # Default to (0, 0, 0) if no primary axis is found

    # Count axes where IK locking is enabled
    limited_axes_count = sum(limits[0] is False for limits in rotation_limits.values())

    # Single axis limited: likely a hinge
    if limited_axes_count == 1:
        joint_type = "hinge"
    elif limited_axes_count == 2:
        joint_type = "ball" 
    elif limited_axes_count == 3:
        joint_type = "hinge"
    else:
        # All axes allow full rotation or no axes are limited: free joint
        joint_type = "free"

    return joint_type, axis_vector


def get_root_armature():
    return next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)

def get_meshes(armature):
    for obj in armature.children:
        if obj.type != 'MESH':
            continue
        yield obj

def parse_blender_file(armature):
    bone_info = {}
    for bone in armature.data.bones:
        bone_info[bone.name] = BoneInfo(
            head=bone.tail_local,
            axis=vector_to_string(calculate_axis(bone)),
            length=str(bone.length),
            parent=bone.parent.name if bone.parent else None,
            joint_type = infer_joint_type(bone, armature),
            mesh=None
        )

    print(bone_info.keys())
    for obj in get_meshes(armature):
        import ipdb; ipdb.set_trace()
        print(obj.bone, obj.name)
        bone_info[obj.parent_bone] = bone_info[obj.parent_bone]._replace(mesh=obj.name)
    return bone_info

def invert_signs(input_tuple):
    return tuple(-x for x in input_tuple)

def write_bone_hierarchy(bone_name, parent_xml_element, bone_info, worldbody, asset):
    bone = bone_info[bone_name]
    body = ET.SubElement(parent_xml_element, 'body', attrib={'name': bone_name, 'pos': vector_to_string(invert_signs(bone.head)) })
    if bone.joint_type:
        joint_attribs = {'name': f"{bone_name}_joint", 'type': bone.joint_type[0], 'pos': "0 0 0", 'axis': vector_to_string(bone.joint_type[1])}
        ET.SubElement(body, 'joint', attrib=joint_attribs)
        inertial = ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    
    if bone.mesh:
        mesh_export_path = os.path.join(export_directory, f"{bone.mesh}.stl")
        export_mesh_as_stl(bone.mesh, mesh_export_path)
        ET.SubElement(asset, 'mesh', attrib={'file': f"{bone.mesh}.stl"})
        ET.SubElement(body, "geom", pos=vector_to_string(bone.head), mesh=bone.mesh)

    for child_bone_name in bone_info:
        if bone_info[child_bone_name].parent == bone_name:
            write_bone_hierarchy(child_bone_name, body, bone_info, worldbody, asset)

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_root_bones(bone_info):
    # Find root bones (bones without a parent) and start the hierarchy from each
    return [bone_name for bone_name, bone in bone_info.items() if bone.parent is None]

def write_mujoco_file(bone_info):
    mujoco, worldbody, asset = create_mujoco_xml()
    for root_bone_name in get_root_bones(bone_info):
        write_bone_hierarchy(root_bone_name, worldbody, bone_info, worldbody, asset)
    return mujoco

def final_write(xml_content):
    with open(mujoco_file_path, 'w') as file:
        file.write(xml_content)

if __name__ == "__main__":
    if "bpy" in sys.modules:
        final_write(prettify_xml(write_mujoco_file(parse_blender_file(get_root_armature()))))
    else:
        subprocess.run([blender_executable_path, '-b', blender_file_path, '--python', __file__])

