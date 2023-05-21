{ buildPythonPackage
, fetchFromGitHub
, absl-py
, contextlib2
, pyyaml
, mock
}:

buildPythonPackage rec {
  name = "ml_collections";
  src = fetchFromGitHub {
    owner = "google";
    repo = "ml_collections";
    rev = "v0.1.1";
    hash = "sha256-On5+OuVIKdP95eLiyT5Cm126KFCOaxjb7bJFkFGjPR0=";
  };
  propagatedBuildInputs = [
    absl-py
    contextlib2
    pyyaml
    mock
  ];

  postInstall = ''
    # Remove the conflicting file(s).
    rm -f $out/lib/python*/site-packages/docs/__pycache__/conf.cpython-310.pyc
  '';

}
