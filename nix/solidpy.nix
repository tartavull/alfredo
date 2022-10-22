with import <nixpkgs> {};
with pkgs.python3Packages;

buildPythonPackage rec {
  name = "solidpy";
  src = fetchFromGitHub {
          owner = "100";
          repo = "Solid";
          rev = "f38ca4906b7a253bfbb74f271229625d0f1df175";
          hash = "sha256-1foBoRkCQieGAMQHGKoBu5MuCCv663BAIOZqpq9has4=";
  };
  propagatedBuildInputs = [ pytest numpy pkgs.libsndfile ];
}
