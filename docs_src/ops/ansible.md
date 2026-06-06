# Remote Linux Hosts (Ansible)

For operational environments where forecasts are run on dedicated bare-metal servers or persistent VM clusters, you can automate host setup and deployment using **Ansible**.

This guide provides an Ansible playbook to install system build tools, compile WeatherGraph from source, and configure operational crons.

---

## 1. Inventory Configuration

Define your operational hosts in a `hosts.ini` inventory file:

```ini
[forecast_servers]
forecast-srv-01.local ansible_host=192.168.1.50 ansible_user=deploy
forecast-srv-02.local ansible_host=192.168.1.51 ansible_user=deploy
```

---

## 2. The Deployment Playbook

Below is a complete playbook (`deploy_weathergraph.yml`) that configures a remote Ubuntu host, installs compilation packages, clones the repository, builds the C++ backend, and downloads the normalization weights:

```yaml
---
- name: Deploy WeatherGraph to Forecast Servers
  hosts: forecast_servers
  become: yes
  vars:
    project_root: "/opt/weathergraph"
    python_version: "python3.11"
    weights_url: "https://assets.wanderspool.org/models/weathergraph_weights.pkl"

  tasks:
    - name: Install System Compilation Packages
      apt:
        name:
          - build-essential
          - cmake
          - patchelf
          - git
          - "{{ python_version }}"
          - "{{ python_version }}-dev"
          - "{{ python_version }}-venv"
        state: present
        update_cache: yes

    - name: Ensure Project Root Directory Exists
      file:
        path: "{{ project_root }}"
        state: directory
        owner: "{{ ansible_user }}"
        group: "{{ ansible_user }}"
        mode: '0755'

    - name: Clone WeatherGraph Repository
      git:
        repo: "https://github.com/Wanderspool/WeatherGraph.git"
        dest: "{{ project_root }}"
        version: "main"
      become: no

    - name: Create Python Virtual Environment
      command: "python3 -m venv {{ project_root }}/.venv"
      args:
        creates: "{{ project_root }}/.venv"
      become: no

    - name: Upgrade Pip in Virtual Environment
      pip:
        name: pip
        state: latest
        virtualenv: "{{ project_root }}/.venv"
      become: no

    - name: Compile C++ core and Install Package
      pip:
        name: "{{ project_root }}"
        state: present
        virtualenv: "{{ project_root }}/.venv"
        extra_args: "-e"
      become: no

    - name: Create Weights Cache Directory
      file:
        path: "{{ project_root }}/data"
        state: directory
        owner: "{{ ansible_user }}"
        group: "{{ ansible_user }}"
        mode: '0755'

    - name: Download Model Weight Archive
      get_url:
        url: "{{ weights_url }}"
        dest: "{{ project_root }}/data/model.pkl"
        mode: '0644'
      become: no
```

### Running the Playbook
To run this setup across your inventory:

```bash
ansible-playbook -i hosts.ini deploy_weathergraph.yml
```

---

## 3. Automating Operational Forecasts

To automate running a daily forecast rollout at 06:00 UTC, you can deploy a systemd timer on the remote host:

### Systemd Service (`/etc/systemd/system/weathergraph-forecast.service`)
```ini
[Unit]
Description=Daily WeatherGraph Forecast Rollout
After=network.target

[Service]
Type=oneshot
User=deploy
WorkingDirectory=/opt/weathergraph
ExecStart=/opt/weathergraph/.venv/bin/weathergraph forecast \
  --model-path /opt/weathergraph/models/weather_gnn.onnx \
  --weights-dir /opt/weathergraph/data \
  --data-source ecmwf_open \
  --steps 40 \
  --output-format netcdf4 \
  --output-path /var/weather/forecasts/%d
```

### Systemd Timer (`/etc/systemd/system/weathergraph-forecast.timer`)
```ini
[Unit]
Description=Run WeatherGraph Daily

[Timer]
OnCalendar=*-*-* 06:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start the automated timer:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now weathergraph-forecast.timer
```