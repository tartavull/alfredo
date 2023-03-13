{ protolint
, pre-commit-hooks
}:

{
  src = ./.;
  hooks = {
    prettier.enable = true;
    statix.enable = true;
    black.enable = true;
    isort.enable = true;
    nixpkgs-fmt.enable = true;
    trailing-whitespace = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/trailing-whitespace-fixer";
      types = [ "text" ];
    };
    end-of-file-fixer = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/end-of-file-fixer";
      types = [ "text" ];
    };
    check-added-large-files = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/check-added-large-files";
      types = [ "text" ];
    };
    check-merge-conflict = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/check-merge-conflict";
      types = [ "text" ];
    };
    mixed-line-ending = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/mixed-line-ending";
      types = [ "text" ];
    };
    check-yaml = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/check-yaml";
      types = [ "text" ];
      files = "\\.yaml$";
    };
    check-xml = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/check-xml";
      types = [ "text" ];
      files = "\\.xml$";
    };
    check-json = {
      enable = true;
      entry = "${pre-commit-hooks}/bin/check-json";
      types = [ "text" ];
      files = "\\.json$";
    };
    protolint = {
      enable = true;
      entry = "${protolint}/bin/protolint lint -fix";
      types = [ "text" ];
      files = "\\.proto$";
    };
  };
}
