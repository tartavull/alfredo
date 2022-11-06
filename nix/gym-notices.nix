{ lib
, buildPythonPackage
, fetchFromGitHub
}:

buildPythonPackage rec {
  pname = "gym-notices";
  version = "0.0.8";

  src = fetchFromGitHub {
    owner = "Farama-Foundation";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-oyck0xU21eFR2vGZfqt0WDd/GkHCqyQaxGMwi2OzUM8=";
  };

  meta = with lib; {
    license = licenses.mit;
    maintainers = with maintainers; [ tartavull ];
  };
}
