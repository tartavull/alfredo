{
  description = "Alfredo: relentlessly learning, persistently failing, but never surrendering.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    nixos.url = "github:nixos/nixpkgs/nixos-22.11";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixos";
    };
    pre-commit.url = "github:cachix/pre-commit-hooks.nix";
  };

  outputs = { self, nixpkgs, utils, nixos, nixos-generators, pre-commit, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      callPackage = pkgs.lib.callPackageWith (pkgs // pkgs.python3Packages);
      snappy = callPackage ./nix/snappy.nix { };
      ml-dtypes = callPackage ./nix/ml-dtypes.nix { };
      dm_env = callPackage ./nix/dm_env.nix { };
      pytinyrenderer = callPackage ./nix/pytinyrenderer.nix { };
      trimesh = callPackage ./nix/trimesh.nix { };
      ml_collections = callPackage ./nix/ml_collections.nix { };
      tensorstore = callPackage ./nix/tensorstore.nix { };
    in
    {
      overlays.dev = final: prev: {
        magmaWithCuda11 = prev.magma.override {
          cudaPackages = final.cudaPackages_11_8;
        };
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (python-final: python-prev: {

            pytorch = python-prev.pytorchWithCuda.override {
              magma = final.magmaWithCuda11;
              cudaPackages = final.cudaPackages_11_8;
            };

            jaxlib = python-prev.jaxlib.override {
              cudaPackages = final.cudaPackages_11_8;
              cudaSupport = true;
            };

            jax = python-prev.jax.override {
              inherit (python-final) jaxlib;
            };

            jaxopt = callPackage ./nix/jaxopt.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
            };

            chex = callPackage ./nix/chex.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
            };

            optax = callPackage ./nix/optax.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
              inherit (python-final) chex;
            };

            orbax = callPackage ./nix/orbax.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
            };

            flax = callPackage ./nix/flax.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
              inherit (python-final) orbax;
              inherit (python-final) optax;
            };

            mujoco = callPackage ./nix/mujoco.nix {
              glfw-py = pkgs.python3Packages.glfw;
            };

            brax = callPackage ./nix/brax.nix {
              inherit (python-final) jax;
              inherit (python-final) jaxlib;
              inherit (python-final) jaxopt;
              inherit (python-final) optax;
              inherit (python-final) flax;
              inherit (python-final) mujoco;
              inherit dm_env;
              inherit pytinyrenderer;
              inherit trimesh;
            };

            alfredo = callPackage ./nix/alfredo.nix { 
              inherit (python-final) brax;
            };
          })
        ];
      };
    } // utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ self.overlays.dev ];
        };
        python-env = pkgs.python3.withPackages (pyPkgs: with pyPkgs; [
          ipython
          numpy
          pandas
          matplotlib
          torchvision
          pytest
          tqdm
          rich
          networkx
          wandb

          jaxlib
          jax
          brax
          mujoco
        ]);
        name = "alfredo";
      in
      rec {
        devShells.default = pkgs.mkShell {
          inherit name;
          
          LD_LIBRARY_PATH="${pkgs.cudaPackages_11_8.cudatoolkit}/lib";

          packages = [
            python-env
            pkgs.pre-commit
          ];

          shellHooks = ''
            ${checks.pre-commit.shellHook}
          '';
        };

        devShells.deploy = pkgs.mkShell {
          shellHook = ''
            echo "This shell provides google-cloud-sdk, ec2-api-tools and deploy-rs for CGP management, AWS mangement and remote deployment capabilities respectively."
            ${checks.pre-commit.shellHook}
          '';
          packages = [
            pkgs.google-cloud-sdk
            pkgs.deploy-rs
            pkgs.pre-commit
            pkgs.ipython
          ];
        };

        # run `nix flake check`
        checks = {
          pre-commit = pre-commit.lib."${system}".run (import ./nix/pre-commit.nix {
            inherit (pkgs) protolint;
            inherit (pkgs.python3Packages) pre-commit-hooks;
          });
        };

        packages = {
          # can't be build on darwin :/
          gcp = nixos-generators.nixosGenerate {
            system = "x86_64-linux";
            modules = [ ./nix/base.nix ];
            format = "gce";
          };
          aws = nixos-generators.nixosGenerate {
            system = "x86_64-linux";
            modules = [ ./nix/base.nix ] ++ [ (_: { amazonImage.sizeMB = 16 * 1024; }) ];
            format = "amazon";
          };
        };
      }
    );
}
