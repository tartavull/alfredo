# Experimenting with Brax 

The following is a self-contained experiment with Brax and PPO.

## Training

To run as background process:

```
python -u seq_training.py > training.log &
```

To view progress:

```
tail -f training.log
```

## Visualizing Trajectories

```
python vis_traj.py <xml-model-file> <network-parameter-file>
```

eg.

```
python vis_traj.py flatworld/flatworld.xml param-store/A0_param_0
```

note: the filepath to the xml file is relative to the alfredo/scenes.
