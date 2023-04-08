{ config, pkgs, lib, ... }:

{

  # Generic settings
  system.stateVersion = "22.11";

  # NVIDIA configuration
  nixpkgs.config.allowUnfree = true;
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
  ];

  # Service to always fetch latest changes from master in github:tartavull/alfredo
  systemd.user.services.geneticFetcher = {
    enable = true;
    path = with pkgs; [ git ];
    after = [ "NetworkManager-wait-online.service" ];
    serviceConfig = {
      WorkingDirectory = "~";
      ExecStart = ''
        if [ -z "$(ls -A $(pwd)/alfredo)" ]; then
          git clone https://github.com/tartavull/alfredo.git
        else
          git pull main
        fi
      '';
    };
  };
}
