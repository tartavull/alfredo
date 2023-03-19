{
  description = "Genetic Intelligence Main Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit.url = "github:cachix/pre-commit-hooks.nix";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pre-commit, nixos-generators }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        callPackage = pkgs.lib.callPackageWith (pkgs // pkgs.python3Packages // overlay);
        overlay = rec {
          stable-baselines = callPackage ./nix/stable-baselines.nix { };
          box2d-py = callPackage ./nix/box2d-py.nix { };
          core = callPackage ./nix/genetic-intelligence.nix { };
          dm_env = callPackage ./nix/dm_env.nix { };
          pytinyrenderer = callPackage ./nix/pytinyrenderer.nix { };
          #mujoco = callPackage ./nix/mujoco.nix { };
          trimesh = callPackage ./nix/trimesh.nix { };
          brax = callPackage ./nix/brax.nix { };
        };

        core-python = pkgs.python3.withPackages (ps: with ps; [
          ipython
          numpy
          pandas
          matplotlib
          pytorch
          pytest
          tqdm
          rich
          # wandb # test failing

          # only supported on linux
          jaxlib
          jax
          overlay.brax
        ]);


      in
      rec {
        # run `nix flake check`
        checks = {
          pre-commit = pre-commit.lib."${system}".run (import ./nix/pre-commit.nix {
            inherit (pkgs) protolint;
            inherit (pkgs.python3Packages) pre-commit-hooks;
          });
        };

        # to run a shell with all packages type `nix develop`
        # This shell only works on Linux
        devShells.default = pkgs.mkShell {
          shellHook = ''
            echo " ---------------------------------"
            echo "| Welgome to Genetic Intelligence |"
            echo " ---------------------------------"
            ${checks.pre-commit.shellHook}
          '';
          packages = [
            core-python
          ];
        };

        # type `nix develop .#deploy` this shell works on all platforms
        devShells.deploy = pkgs.mkShell {
          shellHook = ''
            echo "This shell provides google-cloud-sdk, ec2-api-tools and deploy-rs for CGP management, AWS mangement and remote deployment capabilities respectively."
            ${checks.pre-commit.shellHook}
          '';
          packages = [
            pkgs.google-cloud-sdk
            unfreepkgs.ec2-api-tools
            pkgs.deploy-rs
          ];
        };
        unfreepkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        packages = {
          # can't be build in darwin
          gcp = nixos-generators.nixosGenerate {
            system = "x86_64-linux";
            modules = [ ./nix/deployer/base.nix ];
            format = "gce";
          };
          aws = nixos-generators.nixosGenerate {
            system = "x86_64-linux";
            modules = [ ./nix/deployer/base.nix ] ++ [ (_: { amazonImage.sizeMB = 16 * 1024; }) ];
            format = "amazon";
          };
          raw = nixos-generators.nixosGenerate {
            system = "x86_64-linux";
            modules = [ ./nix/deployer/base.nix ];
            format = "raw-efi";
          };
        };
      });
}
