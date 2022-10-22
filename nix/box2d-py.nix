with import <nixpkgs> {};
with pkgs.python3Packages;

buildPythonPackage rec {
  name = "box2d-py";
  src = fetchFromGitHub {
          owner = "openai";
          repo = "box2d-py";
          rev = "647d6c66710cfe3ff81b3e80522ff92624a77b41";
          hash = "sha256-TRQd3Du4CkurK7x3nrT8OqzaDzy7Z14wP1TfmHZoaGk=";
  };
  nativeBuildInputs = [ swig ];
  propagatedBuildInputs = [ ];
}
