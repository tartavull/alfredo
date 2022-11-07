{ pkgs ? import <nixpkgs> {} }:


with pkgs.python3Packages;

let
  packages = rec {
    solidpy = callPackage ./solidpy.nix {};
    stable-baselines = callPackage ./stable-baselines.nix {};
    box2d-py = callPackage ./box2d-py.nix {};
    gym-notices =  callPackage ./gym-notices.nix {};
    /* stable baselines depends on gym=0.21 so we can't update to a newer version
    gym = callPackage ./gym.nix {
      gym-notices = gym-notices;
    };
    */
    genetic-intelligence = callPackage ./genetic-intelligence.nix {
        stable-baselines = stable-baselines;
    };
    python = pkgs.python3.withPackages(ps: with ps; [ 
        gym
        solidpy
        stable-baselines
        box2d-py
        gym-notices
        genetic-intelligence

        ipython
        numpy 
        pandas
        matplotlib
        pytorch
        pyglet
        pytest
        tqdm
        rich
        wandb
      ]);
  };
in
  pkgs.mkShell {
      nativeBuildInputs = [
        packages.python
      ];
      shellHook = ''
        cd ..
      '';
  }
