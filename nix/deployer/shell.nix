{ pkgs ? import <nixpkgs> { } }:

with pkgs;

mkShell {

  packages = [
    google-cloud-sdk
    ec2-api-tools
    deploy-rs
  ];

  shellHook = ''
    echo ""
    echo "This shell provides google-cloud-sdk, ec2-api-tools and deploy-rs for CGP management, AWS mangement and remote deployment capabilities respectively."
  '';
}
