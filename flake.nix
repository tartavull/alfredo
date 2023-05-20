{
  description = "Alfredo: relentlessly learning, persistently failing, but never surrendering.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    utils.url = "github:numtide/flake-utils";

    ml-pkgs = {
      url = "github:nixvital/ml-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.utils.follows = "utils";
    };

    pre-commit.url = "github:cachix/pre-commit-hooks.nix";

    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, utils, ml-pkgs, pre-commit, nixos-generators, ... } @inputs: {
    overlays.dev = nixpkgs.lib.composeManyExtensions [
      ml-pkgs.overlays.torch-family
      ml-pkgs.overlays.jax-family

      # You can put your other overlays here, inline or with import. For example
      # if you want to put an inline overlay, uncomment below:
      #
      # (final: prev: {
      #   pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
      #     (python-final: python-prev: {
      #       my-package = ...;
      #     })
      #   ];
      # })
    ];
  } // utils.lib.eachSystem [
    "x86_64-linux"
  ]
    (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ self.overlays.dev ];
        };
      in
      {
        checks = {
          pre-commit = pre-commit.lib."${system}".run (import ./nix/pre-commit.nix {
            inherit (pkgs) protolint;
            inherit (pkgs.python3Packages) pre-commit-hooks;
          });
        };

        devShells.default =
          let
            python-env = pkgs.python3.withPackages (pyPkgs: with pyPkgs; [
              numpy
              pandas
              pytorchWithCuda11
              torchvisionWithCuda11
              # jaxlibWithCuda11
              # jaxWithCuda11
              ipython
              matplotlib
              pytest
              tqdm
              rich
              networkx
              # overlay.brax
              # overlay.mplcursors
              # overlay.alfredo
              # overlay.wandb
            ]);

            name = "alfredo-shell";
          in
          pkgs.mkShell {
            inherit name;

            packages = [
              python-env
            ];

            shellHooks = let pythonIcon = "f3e2"; in ''
              export PS1="$(echo -e '\u${pythonIcon}') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
              echo " -------------------------------------------------------------------------------"
              echo "| Alfredo: relentlessly learning, persistently failing, but never surrendering. |"
              echo " -------------------------------------------------------------------------------"
            '';
          };

        packages = {
          # can't be build on darwin :/
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
