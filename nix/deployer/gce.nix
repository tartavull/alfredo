{ modulesPath, ... }:

{

  imports = [
    "${modulesPath}/virtualisation/google-compute-image.nix"
    ./base.nix
  ];

}
