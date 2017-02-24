# Legion Test Infrastructure

This document describes Legion's regression test infrastructure.

There are two key parts of Legion's test infrastructure:

  * The `test.py` script contains all the logic for running tests.
  * An automated CI framework is used to run this script on every commit.

This separation has the advantage the test suite itself should run
anywhere and ideally produce the same results (though environmental
factors may vary).

## Running Tests Manually

Everything runs through the main test script:

```
./test.py
```

By default, the script will choose an appropriate set of tests and
build options to run. The script will print its configuration at the
start so you can see what tests are enabled. There are two sets of
options that control tests and build flags respectively, which can
either be set on the command line or via environment variables.

Tests:

  * `--test=regent` or `TEST_REGENT`: Regent test suite
  * `--test=legion_cxx` or `TEST_LEGION_CXX`: Legion C++ examples and tests
  * `--test=fuzzer` or `TEST_FUZZER`: Legion fuzzer randomized tests
  * `--test=realm` or `TEST_REALM`: Realm tests
  * `--test=external` or `TEST_EXTERNAL`: Various external applications
  * `--test=private` or `TEST_PRIVATE`: Private external applications
  * `--test=perf` or `TEST_PERF`: [Performance tests](README.perf.md)

Build flags:

  * `--debug` or `DEBUG`: Enable debug mode (disable with `DEBUG=0`)
  * `--use=gasnet` or `USE_GASNET`: Enable GASNet networking
  * `--use=cuda` or `USE_CUDA`: Enable CUDA for NVIDIA GPUs
  * `--use=llvm` or `USE_LLVM`: Enable LLVM support
  * `--use=hdf` or `USE_HDF`: Enable HDF5 I/O support
  * `--use=spy` or `USE_SPY`: Enable Legion Spy
  * `--use=cmake` or `USE_CMAKE`: Enable the CMake build system (vs Makefiles)
  * `--use=rdir` or `USE_RDIR`: Enable RDIR, a plugin for Regent

A note about command-line flags vs environment variables: flags are
exclusive, while variables are additive. That is, the following two
commands will do different things:

```
./test --test=regent
TEST_REGENT=1 ./test.py
```

The first will run **only** Regent tests. The second will run Regent
tests **in addition to** any other tests already enabled (in this
case, whatever defaults are chosen by the script). The same applies to
`--use` and `USE_` options as well.

### How to Add New Tests

## Automated Test Infrastructure

The Legion project currently uses *two* different CI services to run
automated tests: [Travis](https://travis-ci.org/) and [Gitlab
CI](https://about.gitlab.com/gitlab-ci/). Each service has a number of
advantages, so at the moment it makes sense to use both.

Travis:

  * Has better Github integration, which is nice for pull requests, etc.
  * Supports builds on macOS
  * However, Travis tends to be slower and provides a less flexible environment

Gitlab CI:

  * Is more flexible:
      * We can provide our own Docker image(s) for tests
      * We can *simultaneously* use on-premises and cloud-based machines
  * Gitlab generally tends to be faster
  * Has no built-in Github integration (for obvious reasons)

An advantage of both approaches is that they are entirely configured
via text files stored in the Legion repository. Thus there is no
long-term state associated with a local installation of a product such
as Jenkins, and very little operational knowledge required to run
either service.

The configurations are stored in the following files:

  * Travis: `.travis.yml`
  * Gitlab CI: `.gitlab-ci.yml`
      * Docker scripts are in the `docker/` directory

The two formats are similar, but different. Since the actual bulk of
the test suite is stored in the `test.py` script, the contents of the
two `.yml` files amount to essentially setting environment variables
and kicking off the script.

### Gitlab Mirroring

There is one slight wrinkle in the Gitlab CI approach: Gitlab only
understands how to run CI jobs on repos hosted on its own
service. Therefore, to run Gitlab CI on a Github repo, that repo must
first be mirrored onto Gitlab.

Gitlab does provide a mirror feature. However, this only mirrors the
contents of the remote repo about once an hour. This means that any CI
jobs would also have an average latency of about half an hour, which
is unnacceptable.

Instead we do the mirroring ourselves. We accomplish this with an
extremely simple cronjob that runs every two minutes (so average
latency is one minute). The following steps describe the process to
set this up. In normal conditions, it should never be necessary to
touch this, but if something goes wrong (e.g. Sapling loses a hard
drive) it might be necessary to do this over.

 1. Create a new user `gitlab-mirror`:

    ```
    sudo adduser --system gitlab-mirror
    sudo chsh -s /bin/bash gitlab-mirror
    ```

 2. Switch to the user `gitlab-mirror` and initialize the local
    mirrors:

    ```
    git clone --mirror https://github.com/StanfordLegion/legion.git legion.git
    git clone --mirror https://github.com/StanfordLegion/rdir.git rdir.git
    ```

 3. Set up the cronjob:

    ```
    */2 * * * * git -C /home/gitlab-mirror/rdir.git fetch && git -C /home/gitlab-mirror/rdir.git push git@gitlab.com:StanfordLegion/rdir.git --mirror && git -C /home/gitlab-mirror/legion.git fetch && git -C /home/gitlab-mirror/legion.git push git@gitlab.com:StanfordLegion/legion.git --mirror
    ```

### Rebuilding the Docker Image

Docker images are also built with scripts to ensure that the results
are reproducible. To update the Docker image for Gitlab, modify the
file `docker/Dockerfile.gitlab-ci` as desired, then commit the change
to the Legion repository.

**Important:** To avoid constantly rebuilding the Docker image, we use
a separate branch to trigger the build. After changing the build
script, always merge `master` into `docker-snapshot` and push the
latter branch to Github. If you forget this step the automatic build
will not trigger.

While the rebuild process is fully automated, it can take a while. If
you get impatient, you can build and upload the image yourself. *You
will need access to the* `stanfordlegion` *organization on Docker Hub
before you can do this.* The following commands will rebuild the
image:

```
# from the root of the Legion repository
docker build -t stanfordlegion/gitlab-ci -f docker/Dockerfile.gitlab-ci docker
docker login
docker push stanfordlegion/gitlab-ci
```
