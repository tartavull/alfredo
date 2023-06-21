{ buildPythonPackage
, geopy
, joblib
, tqdm
}:

buildPythonPackage rec {
  name = "alfredo";
  src = ../alfredo/.;
  propagatedBuildInputs = [
    geopy
    joblib
    tqdm
  ];
}
