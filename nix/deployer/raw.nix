{ config, lib, options, pkgs, ... }:

{

  imports = [
    ./base.nix
  ];

  boot.loader.grub = {
    device = "nodev";
    efiSupport = true;
    efiInstallAsRemovable = true;
  };

  fileSystems."/boot" = {
    device = "/dev/vda1";
    fsType = "vfat";
  };

  fileSystems."/" = {
    device = "/dev/disk/by-label/nixos";
    autoResize = true;
    fsType = "ext4";
  };

  boot = {
    growPartition = true;
    kernelParams = [ "console=ttyS0" ];
    loader.grub.device = lib.mkDefault "/dev/vda";
    loader.timeout = 0;
    initrd.availableKernelModules = [ "uas" ];
  };

}
