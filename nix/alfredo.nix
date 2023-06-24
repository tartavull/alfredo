{ buildPythonPackage
, geopy
, joblib
, tqdm
, brax
}:

buildPythonPackage rec {
  name = "alfredo";
  src = ../alfredo/.;
  propagatedBuildInputs = [
    geopy
    joblib
    tqdm
    brax
  ];
}
