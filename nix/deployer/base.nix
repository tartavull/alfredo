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
      openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHD6vu2OC+lMAF6mA+Nbn3aCH8/r9EHkNk8OM5QsV6Iw tartavull@gmail.com" ]; # Don't forget to add your ssh public key here!
    };
    root = {
      openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHD6vu2OC+lMAF6mA+Nbn3aCH8/r9EHkNk8OM5QsV6Iw tartavull@gmail.com" ]; # Don't forget to add your ssh public key here!
      hashedPassword = "!";
    };
  };

  # Packages
  environment.systemPackages = with pkgs; [
    google-cloud-sdk
    ec2-api-tools
    deploy-rs
    git
  ];

  # Service to always fetch latest changes from master in github:tartavull/alfredo
  systemd.user.services.geneticFetcher = {
    enable = true;
    path = with pkgs; [ git ];
    after = [ "NetworkManager-wait-online.service" ];
    serviceConfig = { };
  };

}
