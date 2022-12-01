{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/69335c46c48a73f291d5c6f332fb9fe8b8e22b30.tar.gz") { }
}:

with pkgs.python3Packages;

let
  packages = rec {
    solidpy = callPackage ./solidpy.nix { };
    stable-baselines = callPackage ./stable-baselines.nix { };
    box2d-py = callPackage ./box2d-py.nix { };
    gym-notices = callPackage ./gym-notices.nix { };
    /* stable baselines depends on gym=0.21 so we can't update to a newer version
      gym = callPackage ./gym.nix {
      inherit gym-notices;
      };
    */
    core-go = callPackage ./core-go.nix { };
    genetic-intelligence = callPackage ./genetic-intelligence.nix {
      inherit stable-baselines;
      inherit core-go;
    };
    python = pkgs.python3.withPackages (ps: with ps; [
      gym
      solidpy
      stable-baselines
      box2d-py
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
    packages.core-go
    pkgs.act
  ];
  shellHook = ''
    ${(import ./pre-commit.nix).pre-commit-check.shellHook}
    export GICORE=${packages.core-go}/core.so
  '';
}
