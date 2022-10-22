with (import <nixpkgs> {});

let
gi = rec {
  solidpy = import ./solidpy.nix;
  stable-baselines = import ./stable-baselines.nix;
  box2d-py = import ./box2d-py.nix;
  python = python310.withPackages(ps: with ps; [ 
      ipython
      numpy 
      pandas
      matplotlib
      pytorch
      gym
      pyglet
      pytest
      tqdm
      rich
      solidpy
      stable-baselines
      box2d-py
    ]);
};

in
pkgs.mkShell {
    nativeBuildInputs = [
      gi.python
    ];
    shellHook = ''
    '';
}
