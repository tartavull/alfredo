{ config, pkgs, lib, ... }:

{

  # Generic settings
  system.stateVersion = "22.11";

  # NVIDIA configuration
  nixpkgs.config.allowUnfree = true;
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.opengl.enable = true;
  nix.settings.experimental-features = [ "nix-command" "flakes" ];

  nix.nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

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
