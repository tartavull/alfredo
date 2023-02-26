{
  description = "Genetic Intelligence Main Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        # To import a flake module
        # 1. Add foo to inputs
        # 2. Add foo as a parameter to the outputs function
        # 3. Add here: foo.flakeModule

      ];
      systems = [ "x86_64-linux" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, system, ... }: rec {
        # Per-system attributes can be defined here. The self' and inputs'
        # module parameters provide easy access to attributes of the same
        # system.

        packages.callPackage = pkgs.lib.callPackageWith (pkgs // pkgs.python3Packages);
        packages.core-go = packages.callPackage ./nix/core-go.nix { };


        packages.python = pkgs.python3.withPackages (ps: with ps; [
          ipython
          numpy
          pandas
          matplotlib
          pytorch
          pytest
          tqdm
          rich
          wandb
          #pyglet # not available in aarch64-darwin

          (packages.callPackage ./nix/stable-baselines.nix { })
          (packages.callPackage ./nix/solidpy.nix { })
          (packages.callPackage ./nix/box2d-py.nix { })
          # (packages.callPackage ./nix/gym-notices.nix {}) not necessary in python3.10
          # (callPackage ./nix/genetic-intelligence.nix { core-go = packages.core-go; }) # not needed right now
        ]);

        # to run a shell with all packages type `nix develop`
        packages.default = pkgs.mkShell {
          shellHook = ''
            echo " ---------------------------------"
            echo "| Welgome to Genetic Intelligence |"
            echo " ---------------------------------"
            ${(import ./nix/pre-commit.nix).pre-commit-check.shellHook}
            export GICORE=${packages.core-go}/core.so
          '';
          packages = [
            packages.python
            packages.core-go
          ];
        };

        # To execute python directly type `nix run .#python`
        apps.perSystem.python = {
          type = "app";
          program = "${packages.ipython}";
        };

      };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.

      };
    };
}
