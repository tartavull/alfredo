{ config, pkgs, lib, ... }:

{
  imports = [
    "${builtins.fetchTarball {
      url = "https://github.com/nix-community/home-manager/archive/release-22.11.tar.gz";
      sha256 = "1cp2rpprcfl4mjsrsrpfg6278nf05a0mpl3m0snksvdalfmc5si5";
    }}/nixos"
  ];

  # Generic settings
  system.stateVersion = "22.11";

  # NVIDIA configuration
  nixpkgs.config = {
    allowUnfree = true;
    cudaSupport = true;
  };

  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.opengl.enable = true;
  nix.settings.experimental-features = [ "nix-command" "flakes" ];


  # Users
  users.users = {
    dev = {
      isNormalUser = true;
      extraGroups = [ "wheel" ];
    };
    root = {
      hashedPassword = "!";
    };
  };


  # Packages
  environment.systemPackages = with pkgs; [
    git
    vim
    cachix
  ];

  nix.settings = {
    substituters = [
      "https://tartavull.cachix.org"
    ];
    trusted-public-keys = [
      "tartavull.cachix.org-1:xxmUheA3nzwan59bFhfKEShnPeDXMeii+sWHnbq8PsQ="
    ];
  };

  # Copy the local folder using home-manager
  home-manager.users.dev = {
    home.stateVersion = "22.11"; # add this line
    /*
      home.file.alfredo = {
      source = "${builtins.getEnv "PWD"}/../.";
      recursive = true;
      };
    */
  };
  /*

    # Systemd service to build the devShell
    systemd.user.services.buildDevShell = {
    enable = true;
    after = [ "home-manager.service" ];
    serviceConfig = {
      WorkingDirectory = "/home/username/foldername";
      ExecStart = "${pkgs.nix}/bin/nix develop";
    };
    };
  */
}
