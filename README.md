# _bloby_
Package that performs blob detection on 3D TIF Stacks <br/>

## Sytem Requirements

The recommended way to use this package is to install [Docker](https://store.docker.com/search?offering=community&type=edition).
Docker is currently available on OS X El Capitan 10.11 and newer macOS releases, the following Ubuntu versions: Zesty 17.04
(LTS), Yakkety 16.10, Xenial 16.04 (LTS), Trusty 14.04 (LTS), and Windows 10.

### Software Dependencies (with version numbers)

The only software dependency needed if using the recommended method is Docker. The following dependencies are included in the Docker Image.

Python depedencies:

colorama --- 0.3.9<br/>
scikit_image --- 0.13.1<br/>
scipy --- 1.0.0<br/>
numpy --- 1.13.1<br/>
requests --- 2.18.4<br/>
intern --- 0.9.4<br/>
tifffile --- 0.12.1<br/>
tqdm --- 4.19.5<br/>
matplotlib --- 2.1.0<br/>
progressbar2 --- 3.34.3<br/>
scikit_learn --- 0.19.1<br/>
pyfiglet --- 0.7.5<br/>

### Versions tested on
We have tested the Docker image and build on macOS High Sierra (on MacBook Pro with 2.9 GHz Intel Core i7 and 16 GB RAM) and Ubuntu Xenial 16.04.3 LTS.

## Installation Guide

Once Docker is installed on your machine, pull the `srivathsapv/bloby` image from Docker Hub [here](https://hub.docker.com/r/srivathsapv/bloby) as follows: <br/>

```
docker pull srivathsapv/bloby
```

It will typically take around 3 minutes to pull the entire Docker image.

## Demo

### Instructions to run on data

Create a `.docker-env` file and add your `BOSS_TOKEN` value as follows. This is needed to upload the detected centroids to BOSS
for visualization

```
BOSS_TOKEN=<your_boss_token>
```

In order to use the functionality built into this Docker image, you need to run the Docker image:

```
docker run -p 3000:3000 --env-file .docker-env srivathsapv/bloby
```

This should print a link to the terminal console that looks like this: <br/>

```
http://0.0.0.0:3000/?token=SOME_TOKEN
```

Go to this link in your browser by copying and pasting it. <br/>

Next, click on `Package Usage.ipynb`. Once the notebook opens, you can run all cells by clicking on 'Cell' and then 'Run All'.

The expected run time for this demo is < 10 seconds.

### Expected Output

You should see a message showing the successful upload to BOSS with a URL!

## Congrats, you've succesfully run _bloby_!

## Other resources:

* [Blob detection algorithm pseudocode](https://github.com/NeuroDataDesign/bloby/wiki/Detection-Algorithm-Pseudocode)
* [Dev docs](https://neurodatadesign.github.io/bloby/)
