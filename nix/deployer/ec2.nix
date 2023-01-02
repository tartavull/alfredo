{ modulesPath, ... }:

{

  imports = [
    "${modulesPath}/virtualisation/amazon-image.nix"
    ./base.nix
  ];
  ec2.hvm = true;

}
