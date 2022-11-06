{ lib
, buildPythonPackage
, fetchFromGitHub
, numpy
, cloudpickle
, gym-notices
}:

buildPythonPackage rec {
  pname = "gym";
  version = "0.26.2";

  src = fetchFromGitHub {
    owner = "openai";
    repo = pname;
    rev = "v${version}";
    url = "https://github.com/openai/gym/archive/refs/tags/${version}.tar.gz";
    hash = "sha256-uJgm8l1SxIRC5PV6BIH/ht/1ucGT5UaUhkFMdusejgA=";
  };

  propagatedBuildInputs = [
    cloudpickle
    numpy
    gym-notices
  ];

  # The test needs MuJoCo that is not free library.
  doCheck = false;

  pythonImportsCheck = [ "gym" ];

  meta = with lib; {
    description = "A toolkit for developing and comparing your reinforcement learning agents";
    homepage = "https://gym.openai.com/";
    license = licenses.mit;
    maintainers = with maintainers; [ hyphon81 ];
  };
}
