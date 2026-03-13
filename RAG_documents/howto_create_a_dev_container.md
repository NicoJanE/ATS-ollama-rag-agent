---
layout: default_c
RefPages:
 - howto_create_a_dev_container
--- 

# Rust – Docker Setup & Usage Guide

![MIT License](https://img.shields.io/badge/License-MIT-green) ![Commercial Services Available](https://img.shields.io/badge/Services-Optional-blue)

## 🛠️ Basic Setup

This guide contains instructions for creating, starting, and customizing a Rust Docker container. To create and start the basic Rust container, execute the following command from the **RustService** directory in Docker Desktop:

<pre class="nje-cmd-one-line">
docker-compose -f compose_rust_cont.yml up -d --build --force-recreate
</pre>

### 🛠️ Web Rocket Setup (Optional)

You can add one or more **Rocket** web framework projects. This creates a new project directory and installs the required libraries.

1. First, set an environment variable to indicate the project name (which will be used as the directory name). On a Windows host, run:

<pre class="nje-cmd-one-line">
$env:PRJ_NAME_ARG="name_of_project_directory"
</pre>

> *Note:*{: style="color: orange;font-size:13px; "} <br>
> <small> If you don't set this variable, a default will be used, taken from the file `.env` (`default_project_name`).* </small>

2. Then, execute the add-on compose file:

<pre class="nje-cmd-one-line">
docker-compose -f compose_rocket_prj_addon.yml up -d --build --force-recreate
</pre>

>*Note:*{: style="color: orange;font-size:13px; "} <br>
> <small>Your project is created in `/app/[project_name]`. If the environment variable is not set, a default name is used.* </small>

3. Open a CLI in the container, navigate to the newly created project directory (`project_name`), and execute these commands to **build** and **run** the project:

<pre class="nje-cmd-multi-line">
cargo build --release    # Builds the project
cargo run --release     # Runs the Rocket server
</pre>

### ✅ Setup Result

After this, you can open a browser on the host and navigate to: [http://localhost:8000/](http://localhost:8000/)

### 📚 Notes and Things to Be Aware of

- **Project files are created in the container (`/app`)**, not on the host. There is a bind mount on the container: `/host/workdir`. On the host, you find it in: `/workdir`. You can use this to copy the projects here, so they are available on the host.
- The new project directory (see step 1) will contain a `rocket.toml` file that holds the web settings. One of these settings is:
  - `address = "0.0.0.0"` — this must be set to `0.0.0.0` to ensure the host can access the website.
- Another important setting is the port. If you change this, make sure to also update the `compose_rust_cont.yml` file. Other settings are available as comments in the file (not all are documented).
- The config file and the template main are copied from the host in the directory: `RustService\Rocket_customized`.
- When running more than one web application on the same server, make sure to use another web port in the `rocket.toml` file and in the Docker Compose files.
- Use `$env:PRJ_NAME_ARG` in the PowerShell host command box to test the new directory that will be created.

<span style="color: #6d757dff; font-size: 13px; font-style: italic;"> 
<i><b>License</b><br>This file is part of: **Rust Template Stack**  Copyright (c) 2025-2026 Nico Jan Eelhart.This repository is [MIT licensed](MIT-license.md) and free to use. For optional commercial support, customization, training, or long-term maintenance, see [COMMERCIAL.md](COMMERCIAL.md).</i>
</span>

<br><br>
<center>─── ✦ ───</center>
