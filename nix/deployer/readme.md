All images contain a systemd service that upn boot fetches the latest changes from master in github:tartavull/genetic-intelligence and places it in the dev user home directory.

## Image building

### GCE

1. Add your ssh public key to users `dev` and `root` in `base.nix`.
1. Enter development shell containing `google-cloud-sdk` by navigating to the directory containing `flake.nix` and running `nix develop`.
1. Authenticate with `gcloud auth login`
   If you've already configured your project and created your bucket, skip to step 6.
1. Create and configure your gcloud project:
1. Create a project with `gcloud projects create <project-name>`
1. Set newly created project as default project `gcloud config set project <project-name>`
1. List billing accounts `gcloud alpha billing accounts list`
1. Link your project to the desired billing account `gcloud alpha billing accounts projects link <project-name> --billing-account=ZZZZZ-ZZZZZZZ-ZZZZZZ`
1. Create and configure the storage bucket for uploading OS images to:
1. Create storage bucket `gsutil mb gs://genetic-nixos-imgs`
1. Set permissions for you storage bucket `gsutil iam ch allUsers:objectViewer gs://genetic-nixos-imgs` (Optional: this sets your storage bucket to publically viewable)
1. Build and upload your GCE image
1. Build image with `nix build .#gcp`
1. Upload image with `gsutil cp "result/<image-filename>" "gs://<bucket-name>/<image-filename>"`
1. Configure image policies:
1. `gsutil acl ch -u AllUsers:R "gs://<bucket-name>/<image-filename>"`
1. `gcloud compute images create "<img-id>" --source-uri "gs://<bucket-name>/<image-filename>" --family="nixos-image-<nixos-version>"`
   <img-id> is the image filename without the .raw.tar.xz extension, and <nixos-version> is currently set to 22-11, if you'd like 23-05, change the nixpkg flake input to "github:nixos/nixpkgs/nixos-unstable".
1. `gcloud compute images add-iam-policy-binding "<img-id>" --member='allAuthenticatedUsers' --role='roles/compute.imageUser'`

That concludes project configuration, now you can create your VM either on the online dashboard or with the `gcloud beta compute` utility with your newly created image. **!IMPORTANT!** Make sure to set `--metadata=enable-oslogin=true` and `--tags=allow-ssh` so that you can access your VM remotely.

Start and stop your node with `gcloud compute instances start <node-name>` and `gcloud compute instances stop <node-name>` respectively. Standard ssh works to access it remotely.

### AWS

I was unable to create an account with AWS, so I am unable to provide instructions on how to configure your project and upload your built image. Nonetheless, building the image is as easy as running `nix build .#aws`. The shell accessed with `nix develop` provides the `ec2-api-tools` package, to interact with the AWS api.

### Generic server

1. Build raw image with `nix build .#raw`
1. `dd` the resulting image to a disk, and boot from it.

Note that this image is made to be compatible with efi enabled devices, and will not boot from a legacy bios enabled machine. First boot will likely take a while, as the image will be expanding to the drive's size.

## Remote management with deploy-rs

If you'd like to modify and push your changes to your nodes, deploy-rs is used to do so. Changes to `ec2.nix`, `gce.nix` and `raw.nix` only apply to AWS, GCP and generic nodes respectively. Changes to `base.nix` apply to all three nodes. To push changes to any desired node, first make sure to add the node's accessible IP address to the `hostname` field in the `# Deployment configuration` section of the `flake.nix` file, and simply run `deploy .#gce-node`, `deploy .#aws-ec-node`, or `deploy .#geenric-node`.
