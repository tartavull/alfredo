{
  description = "Genetic Intelligence Main Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pre-commit.url = "github:cachix/pre-commit-hooks.nix";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
      ];
      systems = [ "x86_64-linux" "aarch64-darwin" ];

      perSystem = { config, self', inputs', pkgs, pre-commit, system, ... }: rec {
        # Per-system attributes can be defined here. The self' and inputs'
        # module parameters provide easy access to attributes of the same
        # system.
        checks = {
          pre-commit = inputs.pre-commit.lib."${system}".run {
            src = ./.;
            hooks = {
              prettier.enable = true;
              statix.enable = true;
              black.enable = true;
              isort.enable = true;
              nixpkgs-fmt.enable = true;
              trailing-whitespace = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer";
                types = [ "text" ];
              };
              end-of-file-fixer = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/end-of-file-fixer";
                types = [ "text" ];
              };
              check-added-large-files = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-added-large-files";
                types = [ "text" ];
              };
              check-merge-conflict = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-merge-conflict";
                types = [ "text" ];
              };
              mixed-line-ending = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/mixed-line-ending";
                types = [ "text" ];
              };
              check-yaml = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-yaml";
                types = [ "text" ];
                files = "\\.yaml$";
              };
              check-xml = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-xml";
                types = [ "text" ];
                files = "\\.xml$";
              };
              check-json = {
                enable = true;
                entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-json";
                types = [ "text" ];
                files = "\\.json$";
              };
              protolint = {
                enable = true;
                entry = "${pkgs.protolint}/bin/protolint lint -fix";
                types = [ "text" ];
                files = "\\.proto$";
              };
            };

          };
        };

        packages.callPackage = pkgs.lib.callPackageWith (pkgs // pkgs.python3Packages // packages.ourPackages);
        packages.ourPackages = {
          stable-baselines = packages.callPackage ./nix/stable-baselines.nix { };
          solidpy = packages.callPackage ./nix/solidpy.nix { };
          box2d-py = packages.callPackage ./nix/box2d-py.nix { };
          # (packages.callPackage ./nix/gym-notices.nix {}) not necessary in python3.10
          core-go = packages.callPackage ./nix/core-go.nix { };
          core = packages.callPackage ./nix/genetic-intelligence.nix { };
        };

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
          jaxlib
          jax # not available in aarch64-darwin
          #pyglet # not available in aarch64-darwin

          packages.ourPackages.stable-baselines
          packages.ourPackages.solidpy
          packages.ourPackages.box2d-py
          packages.ourPackages.core
        ]);

        # to run a shell with all packages type `nix develop`
        devShells.default = pkgs.mkShell {
          shellHook = ''
            echo " ---------------------------------"
            echo "| Welgome to Genetic Intelligence |"
            echo " ---------------------------------"
            export GICORE=${packages.ourPackages.core-go}/core.so
            ${checks.pre-commit.shellHook}
          '';
          packages = [
            packages.python
            packages.ourPackages.core-go
          ];
        };
      };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.

      };
    };
}
