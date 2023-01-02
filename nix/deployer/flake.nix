{
  description = "Deploy NixOS and Genetic Intelligence to AWS, GCE or a local server.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    deploy-rs = {
      url = "github:serokell/deploy-rs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, nixos-generators, deploy-rs, ... }: {
    # Development shell, called with `nix develop`
    devShells = nixpkgs.lib.genAttrs nixpkgs.lib.platforms.unix (
      system: {
        default = import ./shell.nix {
          pkgs = import nixpkgs {
            inherit system;
            config = { allowUnfree = true; };
          };
        };
      }
    );

    # Image generators
    packages.aarch64-darwin = {
      gcp = nixos-generators.nixosGenerate {
        system = "x86_64-linux";
        modules = [ ./base.nix ];
        format = "gce";
      };
      aws = nixos-generators.nixosGenerate {
        system = "x86_64-linux";
        modules = [ ./base.nix ] ++ [ (_: { amazonImage.sizeMB = 16 * 1024; }) ];
        format = "amazon";
      };
      raw = nixos-generators.nixosGenerate {
        system = "x86_64-linux";
        modules = [ ./base.nix ];
        format = "raw-efi";
      };
    };

    # System configuration
    nixosConfigurations = {
      gce-node = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          ./gce.nix
        ];
      };
      aws-ec-node = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          ./ec2.nix
        ];
      };
      generic-node = nixpkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          ./raw.nix
        ];
      };
    };

    # Deployment configuration
    deploy = {
      autoRollback = false;
      magicRollback = false;
      remoteBuild = true; # Set to false if you'd like to build the image locally, then upload to your node.
      nodes = {
        gce-node = {
          hostname = "";
          profiles.system = {
            path = deploy-rs.lib.x86_64-linux.activate.nixos self.nixosConfigurations.gce-node;
          };
        };
        aws-ec-node = {
          hostname = "";
          profiles.system = {
            path = deploy-rs.lib.x86_64-linux.activate.nixos self.nixosConfigurations.aws-ec2-node;
          };
        };
        generic-node = {
          hostname = "";
          profiles.system = {
            path = deploy-rs.lib.x86_64-linux.activate.nixos self.nixosConfigurations.generic-node;
          };
        };
      };
    };
  };
}
