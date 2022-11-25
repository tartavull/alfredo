let 
  pkgs = import (
    fetchTarball "https://github.com/NixOS/nixpkgs/archive/69335c46c48a73f291d5c6f332fb9fe8b8e22b30.tar.gz"
  ) {};
  nix-pre-commit-hooks = import (
    builtins.fetchTarball "https://github.com/cachix/pre-commit-hooks.nix/tarball/master"
  );

in
{
  pre-commit-check = nix-pre-commit-hooks.run {
     src = ../.;
     
     hooks = {
      prettier.enable = true;
      statix.enable = true;
      black.enable = true;
      isort.enable = true;
      nixpkgs-fmt.enable = true;
      trailing-whitespace = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer";
        types = ["text"];
      };
      end-of-file-fixer = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/end-of-file-fixer";
        types = ["text"];
      };
      check-added-large-files = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-added-large-files";
        types = ["text"];
      };
      check-merge-conflict = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-merge-conflict";
        types = ["text"];
      };
      mixed-line-ending = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/mixed-line-ending";
        types = ["text"];
      };
      check-yaml = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-yaml";
        types = ["text"];
        files = "\\.yaml$";
      };
      check-xml = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-xml";
        types = ["text"];
        files = "\\.xml$";
        };
      check-json = {
        enable = true;
        entry = "${pkgs.python3Packages.pre-commit-hooks}/bin/check-json";
        types = ["text"];
        files = "\\.json$";
      };
      protolint = {
        enable = true;
        entry = "${pkgs.protolint}/bin/protolint lint -fix";
        types = ["text"];
        files = "\\.proto$";
      };
     };
  };
}
