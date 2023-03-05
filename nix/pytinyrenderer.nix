{ buildPythonPackage
, fetchFromGitHub
}:

buildPythonPackage rec {
  name = "pytinyrenderer";
  src = fetchFromGitHub {
    owner = "erwincoumans";
    repo = "tinyrenderer";
    rev = "0e8b3fe69f34f0b01683760c3eee29b077410631";
    hash = "sha256-m/iniw7CWy+2rj0+4Q9/MaaTQp04NtBbIGLdHxvRpjU=";
  };
  propagatedBuildInputs = [ ];
}
