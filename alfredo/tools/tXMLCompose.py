import functools
import os
import re
import sys

import xml.etree.ElementTree as ET

def compose_scene(xml_env, xml_agent):
    body_index = {}
    
    env_tree = ET.parse(xml_env)
    agent_tree = ET.parse(xml_agent)

    env_root = env_tree.getroot()
    worldbody = env_root.find('worldbody')

    ag_root = agent_tree.getroot()
    ag_body = ag_root.find('body')
    ag_actuator = ag_root.find('actuator')

    worldbody.append(ag_body)
    env_root.append(ag_actuator)
    
    beautify(env_root)

    scene_xml_string = ET.tostring(env_root, encoding='utf-8')

    return scene_xml_string.decode('utf-8')

def beautify(element, indent='  '):
    queue = [(0, element)]  # (level, element)
    
    while queue:
        level, element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = '\n' + indent * (level+1)  # for child open
        if queue:
            element.tail = '\n' + indent * queue[0][0]  # for child close
        else:
            element.tail = '\n' + indent * (level-1)  # for my close
        
        queue[0:0] = children  # prepend children to process them next

if __name__ == '__main__':
    
    # example usage of compose_scene function
    import alfredo.scenes as scenes
    scene_fp = os.path.dirname(scenes.__file__)
    env_xml_path = f"{scene_fp}/flatworld/flatworld_A1_env.xml"
    
    import alfredo.agents as agents
    agents_fp = os.path.dirname(agents.__file__)
    agent_xml_path = f"{agents_fp}/A1/a1.xml"

    xml_scene = compose_scene(env_xml_path, agent_xml_path)
    print(xml_scene)

    from typing import Tuple
    import jax
    from brax import actuator, base, math
    from brax.envs import PipelineEnv, State
    from brax.io import mjcf
    from etils import epath
    from jax import numpy as jp
    
    sys = mjcf.loads(xml_scene)
    print(sys)
    
