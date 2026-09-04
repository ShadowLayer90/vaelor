"""The Pironman-era ``/api/v1.0`` surface, kept apart from the v2 control plane.

This module owns nothing. Every reading and every write it performs goes
through the ``state`` module handed to :func:`register_legacy_v1_routes`,
because the compatibility surface is driven by the module-level callbacks that
:class:`vaelor.control_plane.VaelorControlPlane` rebinds while the appliance is
running. Resolving them per request, rather than capturing them at import,
keeps a late ``set_read_config`` visible to these handlers.

The whole surface is a temporary compatibility alias. It answers ``410`` unless
``VAELOR_ENABLE_LEGACY_V1=1``, and it must not gain new callers.
"""

from __future__ import annotations

import re
from os import listdir, path, remove

from flask import request
from flask_cors import cross_origin

from .legacy_hardware import get_disks, get_ips
from .runtime_paths import env_value

API_PREFIX = '/api/v1.0'
DEBUG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
AVAILABLE_PIPOWER5_EVENT = [
    "battery_activated",
    "low_battery",
    "power_disconnected",
    "power_restored",
    "power_insufficient",
    "battery_critical_shutdown",
    "battery_voltage_critical_shutdown",
]

__mqtt_connected__ = False


def on_mqtt_connected(client, userdata, flags, rc):
    global __mqtt_connected__
    if rc==0:
        __mqtt_connected__ = True
    else:
        __mqtt_connected__ = False

def get_log_level(line):
    for level in DEBUG_LEVELS:
        if f"[{level}]" in line:
            return level
    return 'INFO'

def _get_log(log_path, name, line_count=100, filter=[], level="INFO"):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
        return False
    if path.exists(f"{log_path}/{name}") == False:
        return False
    with open(f"{log_path}/{name}", 'r') as f:
        lines = f.readlines()
        lines = lines[-line_count:]
        data = []
        for line in lines:
            check = True
            if len(filter) > 0:
                for f in filter:
                    if f in line:
                        break
                else:
                    check = False
            log_level = DEBUG_LEVELS.index(level)
            current_log_level = DEBUG_LEVELS.index(get_log_level(line))
            if current_log_level < log_level:
                check = False
            if check:
                data.append(line)
        return data

def _test_mqtt(config, timeout=5):
    global __mqtt_connected__
    import paho.mqtt.client as mqtt
    from socket import gaierror
    import time
    __mqtt_connected__ = None
    client = mqtt.Client()
    client.on_connect = on_mqtt_connected
    client.username_pw_set(config['username'], config['password'])
    try:
        client.connect(config['host'], config['port'])
    except gaierror:
        return False, "Connection failed, Check hostname and port"
    timestart = time.time()
    while time.time() - timestart < timeout:
        client.loop()
        if __mqtt_connected__ == True:
            return True, None
        elif __mqtt_connected__ == False:
            return False, "Connection failed, Check username and password"
    return False, "Timeout"


def register_legacy_v1_routes(app, state) -> None:
    """Attach the v1 surface to ``app``, reading live globals from ``state``.

    ``state`` is the :mod:`vaelor.control_plane` module itself. Its callback
    globals are replaced after this registration runs, so every handler below
    dereferences them at request time.
    """

    @app.before_request
    def disable_legacy_api_by_default():
        if (
            env_value("VAELOR_ENABLE_LEGACY_V1", "PM_ENABLE_LEGACY_V1", "0") != "1"
            and request.path.startswith(API_PREFIX)
        ):
            return {
                "status": False,
                "error": "Legacy API disabled. Use the authenticated /api/v2 control plane.",
            }, 410

    # host API
    @app.route(f'{API_PREFIX}/get-version')
    @cross_origin()
    def get_version():
        from .version import __version__
        return {"status": True, "data": __version__}

    @app.route(f'{API_PREFIX}/get-device-info')
    @cross_origin()
    def get_device_info():
        return {"status": True, "data": state.__device_info__}

    @app.route(f'{API_PREFIX}/test')
    @cross_origin()
    def test():
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/test-mqtt')
    @cross_origin()
    def test_mqtt():
        host = request.args.get("host")
        port = request.args.get("port")
        username = request.args.get("username")
        password = request.args.get("password")
        mqtt_config = {}
        data = None
        status = True
        error = None
        if host is None:
            status = False
            error = "[ERROR] host not found"
        elif port is None:
            status = False
            error = "[ERROR] port not found"
        elif username is None:
            status = False
            error = "[ERROR] username not found"
        elif password is None:
            status = False
            error = "[ERROR] password not found"
        else:
            mqtt_config['host'] = host
            mqtt_config['port'] = int(host)
            mqtt_config['username'] = username
            mqtt_config['password'] = password
            result = _test_mqtt(mqtt_config)
            data = {
                "status": result[0],
                "error": result[1]
            }
            status = True
        result = {"status": status}
        if status:
            result['data'] = data
        else:
            result['error'] = error
        return result

    @app.route(f'{API_PREFIX}/get-data')
    @cross_origin()
    def get_data():
        try:
            if state.__enable_history__ == False:
                data = state.__data_logger__.get_data()
            else:
                num = request.args.get("n")
                if num is None:
                    num = 1
                else:
                    num = int(num)
                data = state.__db__.get("history", n=num)
            return {"status": True, "data": data}
        except Exception as e:
            return {"status": False, "error": str(e)}

    @app.route(f'{API_PREFIX}/get-history')
    @cross_origin()
    def get_history():
        try:
            if state.__enable_history__ == False:
                data = state.__data_logger__.get_data()
            else:
                num = request.args.get("n")
                if num is None:
                    num = 1
                else:
                    num = int(num)
                data = state.__db__.get("history", n=num)
            return {"status": True, "data": data}
        except Exception as e:
            return {"status": False, "error": str(e)}

    @app.route(f'{API_PREFIX}/get-time-range')
    @cross_origin()
    def get_time_range():
        try:
            if state.__enable_history__:
                start = request.args.get("start")
                end = request.args.get("end")
                key = request.args.get("key")
                data = state.__db__.get_data_by_time_range("history", start, end, key)
                return {"status": True, "data": data}
            else:
                return {"status": False, "error": "History is not enabled"}
        except Exception as e:
            return {"status": False, "error": str(e)}

    @app.route(f'{API_PREFIX}/get-config')
    @cross_origin()
    def get_config():
        return {"status": True, "data": state.__read_config__()}

    @app.route(f'{API_PREFIX}/get-log-list')
    @cross_origin()
    def get_log_list():
        log_files = listdir(state.__log_path__)
        return {"status": True, "data": log_files}

    @app.route(f'{API_PREFIX}/get-log')
    @cross_origin()
    def get_log():
        filename = request.args.get("filename")
        filter = request.args.get("filter")
        level = request.args.get("level")
        lines = request.args.get("lines")
        if filename is None:
            return {"status": False, "error": "[ERROR] file not found"}
        if lines is None:
            lines = 100
        else:
            lines = int(lines)
        if filter is not None:
            filter = filter.split(',')
        else:
            filter = []
        if level is None:
            level = "INFO"
        else:
            if level not in DEBUG_LEVELS:
                return {"status": False, "error": f"[ERROR] level {level} not found"}
        content = _get_log(state.__log_path__, filename, lines, filter, level)
        if content is False:
            return {"status": False, "error": f"[ERROR] file {filename} not found"}
        return {"status": True, "data": content}

    @app.route(f'{API_PREFIX}/get-default-on')
    @cross_origin()
    def get_default_on():
        default_on = state.__db__.get("history", "default_on")
        return {"status": True, "data": default_on}

    # deprecated
    @app.route(f'{API_PREFIX}/get-disk-list')
    @cross_origin()
    def get_disk_list():
        return {"status": True, "data": get_disks()}

    # deprecated
    @app.route(f'{API_PREFIX}/get-network-interface-list')
    @cross_origin()
    def get_network_interface_list():
        interfaces = list(get_ips().keys())
        return {"status": True, "data": interfaces}

    @app.route(f'{API_PREFIX}/set-temperature-unit', methods=['POST'])
    @cross_origin()
    def set_temperature_unit():
        unit = request.json["unit"]
        unit = unit.upper()
        if unit not in ['C', 'F']:
            return {"status": False, "error": f"[ERROR] temperature unit {unit} not found, available units: C, F"}
        state.__on_config_changed__({'system': {'temperature_unit': unit}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-shutdown-percentage', methods=['POST'])
    @cross_origin()
    def set_shutdown_percentage():
        percentage = request.json["shutdown-percentage"]
        state.__on_config_changed__({'system': {'shutdown_percentage': percentage}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-fan-led', methods=['POST'])
    @cross_origin()
    def set_fan_led():
        led = request.json["led"]
        if led not in ['on', 'off', 'follow']:
            return {"status": False, "error": f"[ERROR] led {led} not found, available values: on, off or follow"}
        state.__on_config_changed__({'system': {'gpio_fan_led': led}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-fan-mode', methods=['POST'])
    @cross_origin()
    def set_fan_mode():
        mode = request.json["fan_mode"]
        if not isinstance(mode, int):
            return {"status": False, "error": f"[ERROR] fan mode {mode} not found, available modes: 0, 1, 2, 3, 4, for Alway On, Performance, Cool, Balance, or Silent"}
        if mode < 0 or mode > 4:
            return {"status": False, "error": f"[ERROR] fan mode {mode} not found, available modes: 0, 1, 2, 3, 4, for Alway On, Performance, Cool, Balance, or Silent"}
        state.__on_config_changed__({'system': {'gpio_fan_mode': mode}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-brightness', methods=['POST'])
    @cross_origin()
    def set_rgb_brightness():
        brightness = request.json["brightness"]
        state.__on_config_changed__({'system': {'rgb_brightness': brightness}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-color', methods=['POST'])
    @cross_origin()
    def set_rgb_color():
        color = request.json["color"]
        state.__on_config_changed__({'system': {'rgb_color': color}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-enable', methods=['POST'])
    @cross_origin()
    def set_rgb_enable():
        enable = request.json["enable"]
        state.__on_config_changed__({'system': {'rgb_enable': enable}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-led-count', methods=['POST'])
    @cross_origin()
    def set_rgb_led_count():
        led_count = request.json["led_count"]
        if "rgb_led_count_min" in state.__read_config__()["system"] and led_count < state.__read_config__()["system"]["rgb_led_count_min"]:
            return {"status": False, "error": f"[ERROR] led count {led_count} not found, available led count: >= {state.__read_config__()['system']['rgb_led_count_min']}"}
        state.__on_config_changed__({'system': {'rgb_led_count': led_count}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-style', methods=['POST'])
    @cross_origin()
    def set_rgb_style():
        style = request.json["style"]
        state.__on_config_changed__({'system': {'rgb_style': style}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-speed', methods=['POST'])
    @cross_origin()
    def set_rgb_speed():
        speed = request.json["speed"]
        state.__on_config_changed__({'system': {'rgb_speed': speed}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-enable', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_enable():
        enable = request.json["enable"]
        state.__on_config_changed__({'system': {'rgb_matrix_enable': enable}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-style', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_style():
        style = request.json["style"]
        state.__on_config_changed__({'system': {'rgb_matrix_style': style}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-color', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_color():
        color = request.json["color"]
        state.__on_config_changed__({'system': {'rgb_matrix_color': color}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-color2', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_color2():
        color = request.json["color"]
        state.__on_config_changed__({'system': {'rgb_matrix_color2': color}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-brightness', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_brightness():
        brightness = request.json["brightness"]
        state.__on_config_changed__({'system': {'rgb_matrix_brightness': brightness}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-rgb-matrix-speed', methods=['POST'])
    @cross_origin()
    def set_rgb_matrix_speed():
        speed = request.json["speed"]
        state.__on_config_changed__({'system': {'rgb_matrix_speed': speed}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-debug-level', methods=['POST'])
    @cross_origin()
    def set_debug_level():
        level = request.json["level"]
        state.__on_config_changed__({'system': {'debug_level': level}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-oled-sleep-timeout', methods=['POST'])
    @cross_origin()
    def set_oled_sleep_timeout():
        timeout = request.json["timeout"]
        if not isinstance(timeout, (int, float)) or timeout < 0:
            return {"status": False, "error": f"[ERROR] timeout {timeout} must be a positive number"}
        state.__on_config_changed__({'system': {'oled_sleep_timeout': timeout}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-oled-enable', methods=['POST'])
    @cross_origin()
    def set_oled_enable():
        enable = request.json["enable"]
        if not isinstance(enable, bool):
            return {"status": False, "error": f"[ERROR] enable {enable} not found, available values: True or False"}
        state.__on_config_changed__({'system': {'oled_enable': enable}})
        return {"status": True, "data": "OK"}

    # deprecated
    @app.route(f'{API_PREFIX}/set-oled-disk', methods=['POST'])
    @cross_origin()
    def set_oled_disk():
        disk = request.json["disk"]
        disks = ["total"]
        disks.extend(get_disks())

        if disk is None:
            disk = "total"
        elif disk not in disks:
            return {"status": False, "error": f"[ERROR] disk {disk} not found, available disks: {disks}"}
        state.__on_config_changed__({'system': {'oled_disk': disk}})
        return {"status": True, "data": "OK"}

    # deprecated
    @app.route(f'{API_PREFIX}/set-oled-network-interface', methods=['POST'])
    @cross_origin()
    def set_oled_network_interface():
        interface = request.json["interface"]
        interfaces = ['all']
        interfaces.extend(get_ips().keys())

        if interface is None:
            interface = "eth0"
        elif interface not in interfaces:
            return {"status": False, "error": f"[ERROR] interface {interface} not found, available interfaces: {interfaces}"}
        state.__on_config_changed__({'system': {'oled_network_interface': interface}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-oled-rotation', methods=['POST'])
    @cross_origin()
    def set_oled_rotation():
        rotation = request.json["rotation"]
        if rotation not in [0, 180]:
            return {"status": False, "error": f"[ERROR] rotation {rotation} not found, available values: 0 or 180"}
        state.__on_config_changed__({'system': {'oled_rotation': rotation}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-oled-pages', methods=['POST'])
    @cross_origin()
    def set_oled_pages():
        pages = request.json["pages"]
        if pages is None:
            return {"status": False, "error": "[ERROR] pages not found"}
        for page in pages:
            if page not in state.AVAILABLE_OLED_PAGES:
                return {"status": False, "error": f"[ERROR] page {page} not found, available pages: {state.AVAILABLE_OLED_PAGES}"}
        state.__on_config_changed__({'system': {'oled_pages': pages}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-send-email-on', methods=['POST'])
    @cross_origin()
    def set_send_email_on():
        if "on" not in request.json:
            return {"status": False, "error": "[ERROR] on not found"}
        on = request.json["on"]
        if on is None:
            return {"status": False, "error": "[ERROR] on not found"}
        for item in on:
            if item not in AVAILABLE_PIPOWER5_EVENT:
                return {"status": False, "error": f"[ERROR] on {item} not found, available values: {AVAILABLE_PIPOWER5_EVENT}"}
        state.__on_config_changed__({'system': {'send_email_on': on}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-send-email-to', methods=['POST'])
    @cross_origin()
    def set_send_email_to():
        if "to" not in request.json:
            return {"status": False, "error": "[ERROR] to not found"}
        to = request.json["to"]
        if to is None:
            return {"status": False, "error": "[ERROR] to not found"}
        state.__on_config_changed__({'system': {'send_email_to': to}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-smtp-server', methods=['POST'])
    @cross_origin()
    def set_smtp_server():
        if "server" not in request.json:
            return {"status": False, "error": "[ERROR] server not found"}
        server = request.json["server"]
        if server is None:
            return {"status": False, "error": "[ERROR] server not found"}
        state.__on_config_changed__({'system': {'smtp_server': server}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-smtp-port', methods=['POST'])
    @cross_origin()
    def set_smtp_port():
        if "port" not in request.json:
            return {"status": False, "error": "[ERROR] port not found"}
        port = request.json["port"]
        if not isinstance(port, int) or port <= 0:
            return {"status": False, "error": "[ERROR] port must be a positive integer"}
        state.__on_config_changed__({'system': {'smtp_port': port}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-smtp-email', methods=['POST'])
    @cross_origin()
    def set_smtp_email():
        if "email" not in request.json:
            return {"status": False, "error": "[ERROR] email not found"}
        email = request.json["email"]
        if email is None:
            return {"status": False, "error": "[ERROR] email not found"}
        state.__on_config_changed__({'system': {'smtp_email': email}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-smtp-password', methods=['POST'])
    @cross_origin()
    def set_smtp_password():
        if "password" not in request.json:
            return {"status": False, "error": "[ERROR] password not found"}
        smtp_password = request.json["password"]
        if smtp_password is None:
            return {"status": False, "error": "[ERROR] password not found"}
        state.__on_config_changed__({'system': {'smtp_password': smtp_password}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-smtp-security', methods=['POST'])
    @cross_origin()
    def set_smtp_security():
        if "security" not in request.json:
            return {"status": False, "error": "[ERROR] security not found"}
        security = request.json["security"]
        if security is None:
            return {"status": False, "error": "[ERROR] security not found"}
        if security not in ['none', 'ssl', 'tls']:
            return {"status": False, "error": "[ERROR] security must be 'none', 'ssl' or 'tls'"}
        state.__on_config_changed__({'system': {'smtp_security': security}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/test-smtp', methods=['POST', 'GET'])
    @cross_origin()
    def test_smtp():
        result = state.__test_smtp__()
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            status, error = result[0], result[1]
        elif result:
            status, error = True, ""
        else:
            status, error = False, "SMTP test failed"
        if status:
            return {"status": status, "data": "OK"}
        else:
            return {"status": status, "error": f'{error}'}

    @app.route(f'{API_PREFIX}/set-pipower5-buzz-on', methods=['POST'])
    @cross_origin()
    def set_pipower5_buzz_on():
        if "on" not in request.json:
            return {"status": False, "error": "[ERROR] on not found"}
        on = request.json["on"]
        if on is None:
            return {"status": False, "error": "[ERROR] on not found"}
        for item in on:
            if item not in AVAILABLE_PIPOWER5_EVENT:
                return {"status": False, "error": f"[ERROR] on {item} not found, available values: {AVAILABLE_PIPOWER5_EVENT}"}
        state.__on_config_changed__({'system': {'pipower5_buzz_on': on}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-pipower5-buzzer-volume', methods=['POST'])
    @cross_origin()
    def set_pipower5_buzzer_volume():
        if "volume" not in request.json:
            return {"status": False, "error": "[ERROR] volume not found"}
        volume = request.json["volume"]
        if volume is None:
            return {"status": False, "error": "[ERROR] volume not found"}
        if volume < 0 or volume > 10:
            return {"status": False, "error": "[ERROR] volume must be between 0 and 100"}
        state.__on_config_changed__({'system': {'pipower5_buzzer_volume': volume}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/play-pipower5-buzzer', methods=['POST'])
    @cross_origin()
    def play_pipower5_buzzer():
        if "event" not in request.json:
            return {"status": False, "error": "[ERROR] event not found"}
        event = request.json["event"]
        if event is None:
            return {"status": False, "error": "[ERROR] event not found"}
        if event not in AVAILABLE_PIPOWER5_EVENT:
            return {"status": False, "error": f"[ERROR] event {event} not found, available values: {AVAILABLE_PIPOWER5_EVENT}"}
        state.__play_pipower5_buzzer__(event)
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/clear-history', methods=['POST', 'GET'])
    @cross_origin()
    def clear_history():
        if state.__enable_history__ == False:
            return {"status": False, "error": "History is not enabled"}
        state.__db__.clear_measurement('history')
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/delete-log-file', methods=['POST'])
    @cross_origin()
    def delete_log_file():
        filename = request.json["filename"]
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", filename):
            return {"status": False, "error": "[ERROR] file not found"}
        if path.exists(f"{state.__log_path__}/{filename}") == False:
            return {"status": False, "error": f"[ERROR] file {filename} not found"}
        try:
            remove(f"{state.__log_path__}/{filename}")
            return {"status": True, "data": "OK"}
        except Exception as e:
            return {"status": False, "error": str(e)}

    @app.route(f'{API_PREFIX}/start-ups-power-failure-simulation', methods=['POST', 'GET'])
    @cross_origin()
    def set_ups_vbus_enable():
        import subprocess
        time = request.json["time"]
        print(f"start-ups-blackout-simulation {time}")
        subprocess.Popen(['sudo', 'pipower5', '--power-failure-simulation', f'{time}'])
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/get-ups-power-failure-simulation', methods=['POST', 'GET'])
    @cross_origin()
    def get_ups_blackout_simulation():
        import json
        import os
        import time

        try:
            timeout = request.json["timeout"]
        except:
            timeout = 3

        st = time.time()
        file_path = '/opt/pipower5/blackout_simulation'
        while os.path.exists(file_path + ".lock") and time.time() - st < timeout:
            print("file is being written ...")
            time.sleep(1)
        with open(file_path + '.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"status": True, "data":data}

    @app.route(f'{API_PREFIX}/set-restart-service', methods=['POST'])
    @cross_origin()
    def set_restart_service():
        state.__restart_service__()
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-shutdown', methods=['POST'])
    @cross_origin()
    def set_shutdown():
        state.__log__.info("Shutdown requested")
        state.__shutdown__()
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-reboot', methods=['POST'])
    @cross_origin()
    def set_reboot():
        state.__log__.info("Reboot requested")
        state.__reboot__()
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/set-database-retention-days', methods=['POST'])
    @cross_origin()
    def set_database_retention_days():
        database_retention_days = request.json["days"]
        if database_retention_days is None:
            return {"status": False, "error": "[ERROR] database_retention_days not found"}
        state.__on_config_changed__({'system': {'database_retention_days': database_retention_days}})
        return {"status": True, "data": "OK"}

    @app.route(f'{API_PREFIX}/get-ips')
    @cross_origin()
    def get_ips_endpoint():
        data = state.__get_ip_data__()
        if not data or not data.get('ips'):
            # Fallback: extract IP keys from cached data (old pm_auto)
            raw = state.__read_data__()
            ip_keys = ['ips', 'network_type'] + [k for k in raw if k.startswith('ip_') or k.startswith('mac_')]
            data = {k: raw[k] for k in ip_keys if k in raw}
        return {"status": True, "data": data}
