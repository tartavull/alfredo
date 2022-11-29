{ buildGoModule
}:

buildGoModule {
  name = "core-go";
  version = "0.0.1";
  src = ../src/core-go/.;
  vendorHash = "sha256-pQpattmS9VmO3ZIQUFn66az8GSmB4IvYhTTCFn6SUmo=";
  buildPhase = ''
    go build -buildmode=c-shared -o core.so main.go
    mkdir $out
    mv core.so $out/
    mv core.h $out/
  '';
}
