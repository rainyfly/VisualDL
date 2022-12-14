#!/user/bin/env python
# Copyright (c) 2017 VisualDL Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
import multiprocessing
import os
import re
import sys
import threading
import time
import urllib
import webbrowser

import requests
from flask import Flask
from flask import make_response
from flask import redirect
from flask import request
from flask import Response
from flask import send_file
from flask_babel import Babel

import visualdl.server
from visualdl import __version__
from visualdl.component.inference.fastdeploy_lib import get_start_arguments
from visualdl.component.inference.fastdeploy_server import create_fastdeploy_api_call
from visualdl.component.inference.model_convert_server import create_model_convert_api_call
from visualdl.component.profiler.profiler_server import create_profiler_api_call
from visualdl.server.api import create_api_call
from visualdl.server.api import get_component_tabs
from visualdl.server.args import parse_args
from visualdl.server.args import ParseArgs
from visualdl.server.log import info
from visualdl.server.serve import upload_to_dev
from visualdl.server.template import Template
from visualdl.utils import update_util

SERVER_DIR = os.path.join(visualdl.ROOT, 'server')

support_language = ["en", "zh"]
default_language = support_language[0]

server_path = os.path.abspath(os.path.dirname(sys.argv[0]))
template_file_path = os.path.join(SERVER_DIR, "./dist")
mock_data_path = os.path.join(SERVER_DIR, "./mock_data/")

check_live_path = '/alive'


def create_app(args):  # noqa: C901
    # disable warning from flask
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None

    app = Flask('visualdl', static_folder=None)
    app.logger.disabled = True

    # set static expires in a short time to reduce browser's memory usage.
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 30

    app.config['BABEL_DEFAULT_LOCALE'] = default_language
    babel = Babel(app)
    api_call = create_api_call(args.logdir, args.model, args.cache_timeout)
    profiler_api_call = create_profiler_api_call(args.logdir)
    inference_api_call = create_model_convert_api_call()
    fastdeploy_api_call = create_fastdeploy_api_call()
    if args.telemetry:
        update_util.PbUpdater(args.product).start()

    public_path = args.public_path
    api_path = public_path + '/api'

    def append_query_string(url):
        query_string = ''
        if request.query_string:
            query_string = '?' + request.query_string.decode()
        return url + query_string

    @babel.localeselector
    def get_locale():
        lang = args.language
        if not lang or lang not in support_language:
            lang = request.accept_languages.best_match(support_language)
        return lang

    if not args.api_only:

        template = Template(
            os.path.join(server_path, template_file_path),
            PUBLIC_PATH=public_path,
            BASE_URI=public_path,
            API_URL=api_path,
            TELEMETRY_ID='63a600296f8a71f576c4806376a9245b'
            if args.telemetry else '',
            THEME='' if args.theme is None else args.theme)

        @app.route('/')
        def base():
            return redirect(append_query_string(public_path), code=302)

        @app.route('/favicon.ico')
        def favicon():
            icon = os.path.join(template_file_path, 'favicon.ico')
            if os.path.exists(icon):
                return send_file(icon)
            return 'file not found', 404

        @app.route(public_path + '/')
        def index():
            return redirect(
                append_query_string(public_path + '/index'), code=302)

        @app.route(public_path + '/<path:filename>')
        def serve_static(filename):
            is_not_page_request = re.search(r'\..+$', filename)
            response = template.render(
                filename if is_not_page_request else 'index.html')
            if not is_not_page_request:
                response.set_cookie(
                    'vdl_lng',
                    get_locale(),
                    path='/',
                    samesite='Strict',
                    secure=False,
                    httponly=False)
            return response

    @app.route(api_path + '/<path:method>', methods=["GET", "POST"])
    def serve_api(method):
        data, mimetype, headers = api_call(method, request.args)
        return make_response(
            Response(data, mimetype=mimetype, headers=headers))

    @app.route(api_path + '/profiler/<path:method>', methods=["GET", "POST"])
    def serve_profiler_api(method):
        data, mimetype, headers = profiler_api_call(method, request.args)
        return make_response(
            Response(data, mimetype=mimetype, headers=headers))

    @app.route(api_path + '/inference/<path:method>', methods=["GET", "POST"])
    def serve_inference_api(method):
        if request.method == 'POST':
            data, mimetype, headers = inference_api_call(method, request.form)
        else:
            data, mimetype, headers = inference_api_call(method, request.args)
        return make_response(
            Response(data, mimetype=mimetype, headers=headers))

    @app.route(api_path + '/fastdeploy/<path:method>', methods=["GET", "POST"])
    def serve_fastdeploy_api(method):
        if request.method == 'POST':
            data, mimetype, headers = fastdeploy_api_call(method, request.form)
        else:
            data, mimetype, headers = fastdeploy_api_call(method, request.args)
        return make_response(
            Response(data, mimetype=mimetype, headers=headers))

    @app.route(
        api_path + '/fastdeploy/fastdeploy_client', methods=["GET", "POST"])
    def serve_fastdeploy_create_fastdeploy_client():
        try:
            if request.method == 'POST':
                fastdeploy_api_call('create_fastdeploy_client', request.form)
                request_args = request.form
            else:
                fastdeploy_api_call('create_fastdeploy_client', request.args)
                request_args = request.args
        except Exception as e:
            error_msg = '{}'.format(e)
            return make_response(error_msg)
        args = urllib.parse.urlencode(request_args)
        if args:
            return redirect(
                api_path + "/fastdeploy/fastdeploy_client/app?{}".format(args),
                code=302)
        return redirect(
            api_path + "/fastdeploy/fastdeploy_client/app", code=302)

    @app.route(
        api_path + "/fastdeploy/fastdeploy_client/<path:path>",
        methods=["GET", "POST"])
    def request_fastdeploy_create_fastdeploy_client_app(path: str):
        '''
        Gradio app server url interface. We route urls for gradio app to gradio server.

        Args:
            path(str): All resource path from gradio server.

        Returns:
            Any thing from gradio server.
        '''
        if request.method == 'POST':
            port = fastdeploy_api_call('create_fastdeploy_client',
                                       request.form)
            request_args = request.form
        else:
            port = fastdeploy_api_call('create_fastdeploy_client',
                                       request.args)
            request_args = request.args
        if path == 'app':
            proxy_url = request.url.replace(
                request.host_url.rstrip('/') + api_path +
                '/fastdeploy/fastdeploy_client/app',
                'http://localhost:{}/'.format(port))
        else:
            proxy_url = request.url.replace(
                request.host_url.rstrip('/') + api_path +
                '/fastdeploy/fastdeploy_client/',
                'http://localhost:{}/'.format(port))
        resp = requests.request(
            method=request.method,
            url=proxy_url,
            headers={
                key: value
                for (key, value) in request.headers if key != 'Host'
            },
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False)
        if path == 'app':
            content = resp.content
            if request_args and 'server_id' in request_args:
                server_id = request_args.get('server_id')
                start_args = get_start_arguments(server_id)
                http_port = start_args.get('http-port', '')
                model_name = start_args.get('default_model_name', '')
                content = content.decode()
                try:
                    default_server_addr = re.search(
                        r'"label": "Server address".*?"value": "".*?}',
                        content).group(0)
                    cur_server_addr = default_server_addr.replace(
                        '"value": ""',
                        '"value": "localhost:{}"'.format(http_port))
                    default_model_name = re.search(
                        r'"label": "model name".*?"value": "".*?}',
                        content).group(0)
                    cur_model_name = default_model_name.replace(
                        '"value": ""', '"value": "{}"'.format(model_name))
                    default_model_version = re.search(
                        r'"label": "model version".*?"value": "".*?}',
                        content).group(0)
                    cur_model_version = default_model_version.replace(
                        '"value": ""', '"value": "{}"'.format('1'))
                    if http_port:
                        content = content.replace(default_server_addr,
                                                  cur_server_addr)
                    if model_name:
                        content = content.replace(default_model_name,
                                                  cur_model_name)
                    content = content.replace(default_model_version,
                                              cur_model_version)
                except Exception:
                    pass
                finally:
                    content = content.encode()
        else:
            content = resp.content
        headers = [(name, value) for (name, value) in resp.raw.headers.items()]
        response = Response(content, resp.status_code, headers)
        return response

    @app.route(api_path + '/component_tabs')
    def component_tabs():
        data, mimetype, headers = get_component_tabs(
            api_call,
            profiler_api_call,
            vdl_args=args,
            request_args=request.args)
        return make_response(
            Response(data, mimetype=mimetype, headers=headers))

    @app.route(check_live_path)
    def check_live():
        return '', 204

    return app


def wait_until_live(args: ParseArgs):
    url = 'http://{host}:{port}'.format(host=args.host, port=args.port)
    while True:
        try:
            requests.get(url + check_live_path)
            info('Running VisualDL at http://%s:%s/ (Press CTRL+C to quit)',
                 args.host, args.port)

            if args.host == 'localhost':
                info(
                    'Serving VisualDL on localhost; to expose to the network, use a proxy or pass --host 0.0.0.0'
                )

            if args.api_only:
                info('Running in API mode, only %s/* will be served.',
                     args.public_path + '/api')

            break
        except Exception:
            time.sleep(0.5)
    if not args.api_only and args.open_browser:
        webbrowser.open(url + args.public_path)


def _run(args):
    args = ParseArgs(**args)
    os.system('')
    info('\033[1;33mVisualDL %s\033[0m', __version__)
    app = create_app(args)
    threading.Thread(target=wait_until_live, args=(args, )).start()
    app.run(debug=False, host=args.host, port=args.port, threaded=False)


def run(logdir=None, **options):
    args = {'logdir': logdir}
    args.update(options)
    p = multiprocessing.Process(target=_run, args=(args, ))
    p.start()
    return p.pid


def main():
    args = parse_args()
    if args.get('dest') == 'service':
        if args.get('behavior') == 'upload':
            upload_to_dev(args.get('logdir'), args.get('model'))
    else:
        _run(args)


if __name__ == '__main__':
    main()
