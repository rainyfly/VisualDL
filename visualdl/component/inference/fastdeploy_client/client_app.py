# Copyright (c) 2022 VisualDL Authors. All Rights Reserve.
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
import gradio as gr
import numpy as np

from .http_client_manager import get_metric_data
from .http_client_manager import GrpcClientManager
from .http_client_manager import HttpClientManager
from .http_client_manager import metrics_table_head
from .http_client_manager import metrics_table_head_en
from .visualizer import visualize_text_to_image

_http_manager = HttpClientManager()
_grpc_manager = GrpcClientManager()


def create_gradio_client_app(args):  # noqa:C901
    css = """
          .gradio-container {
              font-family: 'IBM Plex Sans', sans-serif;
          }
          .gr-button {
              color: white;
              border-color: black;
              background: black;
          }
          input[type='range'] {
              accent-color: black;
          }
          .dark input[type='range'] {
              accent-color: #dfdfdf;
          }
          #gallery {
              min-height: 22rem;
              margin-bottom: 15px;
              margin-left: auto;
              margin-right: auto;
              border-bottom-right-radius: .5rem !important;
              border-bottom-left-radius: .5rem !important;
          }
          #gallery>div>.h-full {
              min-height: 20rem;
          }
          .details:hover {
              text-decoration: underline;
          }
          .gr-button {
              white-space: nowrap;
          }
          .gr-button:focus {
              border-color: rgb(147 197 253 / var(--tw-border-opacity));
              outline: none;
              box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
              --tw-border-opacity: 1;
              --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) \
                var(--tw-ring-offset-color);
              --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
              --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
              --tw-ring-opacity: .5;
          }
          .footer {
              margin-bottom: 45px;
              margin-top: 35px;
              text-align: center;
              border-bottom: 1px solid #e5e5e5;
          }
          .footer>p {
              font-size: .8rem;
              display: inline-block;
              padding: 0 10px;
              transform: translateY(10px);
              background: white;
          }
          .dark .footer {
              border-color: #303030;
          }
          .dark .footer>p {
              background: #0b0f19;
          }
          .prompt h4{
              margin: 1.25em 0 .25em 0;
              font-weight: bold;
              font-size: 115%;
          }
    """

    block = gr.Blocks(css=css)

    with block:
        gr.HTML("""
              <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                <div
                  style="
                    display: inline-flex;
                    gap: 0.8rem;
                    font-size: 1.75rem;
                    justify-content: center;
                  "
                >
                <h1>
                FastDeploy Client
                </h1>
                </div>
                <p font-size: 94%">
                The client is used for creating requests to fastdeploy server.
                </p>
              </div>
          """)
        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        server_addr_text = gr.Textbox(
                            label="服务ip",
                            show_label=True,
                            max_lines=1,
                            placeholder="localhost",
                            value=args.request_ip)
                        with gr.Column():
                            server_http_port_text = gr.Textbox(
                                label="推理服务端口",
                                show_label=True,
                                max_lines=1,
                                placeholder="8001",
                                value=args.request_port)
                            port_type_text = gr.Radio(
                                ['http', 'grpc'],
                                value=args.request_port_type,
                                label="端口类型")

                        server_metric_port_text = gr.Textbox(
                            label="性能服务端口",
                            show_label=True,
                            max_lines=1,
                            placeholder="8002",
                            value=args.request_metric)
                    with gr.Row():
                        model_name_text = gr.Textbox(
                            label="模型名称",
                            show_label=True,
                            max_lines=1,
                            placeholder="stable_diffusion",
                            value="stable_diffusion")
                        model_version_text = gr.Textbox(
                            label="模型版本",
                            show_label=True,
                            max_lines=1,
                            placeholder="1",
                            value="1")

            with gr.Box():
                input_column = gr.Column()
                with input_column:
                    gr.Markdown("请输入Prompt文本进行描述")
                    input_text = gr.Textbox(label="文本", max_lines=1000)

                output_column = gr.Column(visible=False)
                with output_column:
                    gr.Markdown("生成图像")
                    output_image = gr.Image(interactive=False)

                with gr.Row():
                    component_submit_button = gr.Button("提交")
                    reset_button = gr.Button("重置")

            with gr.Box():
                with gr.Column():
                    gr.Markdown("服务性能统计（每次提交请求会自动更新数据，您也可以手动点击更新）")
                    output_html_table = gr.HTML(
                        label="metrics",
                        interactive=False,
                        show_label=False,
                        value=metrics_table_head.format('', ''))
                    update_metric_button = gr.Button("更新数据")

            status_text = gr.Textbox(
                label="status",
                show_label=True,
                max_lines=1,
                interactive=False)

        lang_text = gr.Textbox(
            label="lang",
            show_label=False,
            value='zh',
            max_lines=1,
            visible=False
        )  # This text box is only used for divide zh and en page

        # input_column
        # input_text
        # output_column
        # output_image
        # component_submit_button
        # reset_button

        def component_inference(*args):
            server_ip = args[0]
            port = args[1]
            port_type = args[2]
            metric_port = args[3]
            model_name = args[4]
            model_version = args[5]

            input_text_data = args[6]

            server_addr = server_ip + ':' + port
            if port_type == 'http':
                manager = _http_manager
            else:
                manager = _grpc_manager
            try:
                input_metas, output_metas = manager.get_model_meta(
                    server_addr, model_name, model_version)
            except Exception as e:
                return {status_text: str(e)}

            if server_ip and port and model_name and model_version:
                inputs = {}
                for i, input_meta in enumerate(input_metas):
                    if input_text_data:
                        print('input_text_data', input_text_data)
                        inputs[input_meta.name] = np.array(
                            [[input_text_data.encode('utf-8')]],
                            dtype=np.object_)
                    else:
                        return {status_text: "请输入Prompt再进行提交"}
                try:
                    infer_results = manager.infer(server_addr, model_name,
                                                  model_version, inputs)
                    results = {status_text: '推理成功'}
                    for i, (output_name,
                            data) in enumerate(infer_results.items()):
                        try:
                            results[output_image] = visualize_text_to_image(
                                data)
                        except Exception:
                            results[output_image] = None
                            return {status_text: "解析模型生成数据发生错误"}
                    if metric_port:
                        html_table = get_metric_data(server_ip, metric_port,
                                                     'zh')
                        results[output_html_table] = html_table
                    results[output_column] = gr.update(visible=True)
                    return results
                except Exception as e:
                    return {status_text: '发生错误: {}'.format(e)}
            else:
                return {status_text: '请输入服务地址、端口号等数据再进行尝试'}

        def update_metric(server_ip, metrics_port, lang_text):
            if server_ip and metrics_port:
                try:
                    html_table = get_metric_data(server_ip, metrics_port, 'zh')
                    return {
                        output_html_table: html_table,
                        status_text: "成功更新性能数据"
                    }
                except Exception as e:
                    return {status_text: 'Error: {}'.format(e)}
            else:
                return {status_text: '请先设置服务ip和端口号port再进行提交'}

        def clear_contents(*args):
            results = {
                output_column: gr.update(visible=False),
                output_image: None,
                input_text: None
            }
            return results

        component_submit_button.click(
            fn=component_inference,
            inputs=[
                server_addr_text, server_http_port_text, port_type_text,
                server_metric_port_text, model_name_text, model_version_text,
                input_text
            ],
            outputs=[
                output_column, output_image, status_text, output_html_table
            ])

        reset_button.click(
            fn=clear_contents,
            inputs=[],
            outputs=[output_column, output_image, input_text])

        update_metric_button.click(
            fn=update_metric,
            inputs=[server_addr_text, server_metric_port_text, lang_text],
            outputs=[output_html_table, status_text])

    return block


def create_gradio_client_app_en(args):  # noqa:C901
    css = """
          .gradio-container {
              font-family: 'IBM Plex Sans', sans-serif;
          }
          .gr-button {
              color: white;
              border-color: black;
              background: black;
          }
          input[type='range'] {
              accent-color: black;
          }
          .dark input[type='range'] {
              accent-color: #dfdfdf;
          }
          #gallery {
              min-height: 22rem;
              margin-bottom: 15px;
              margin-left: auto;
              margin-right: auto;
              border-bottom-right-radius: .5rem !important;
              border-bottom-left-radius: .5rem !important;
          }
          #gallery>div>.h-full {
              min-height: 20rem;
          }
          .details:hover {
              text-decoration: underline;
          }
          .gr-button {
              white-space: nowrap;
          }
          .gr-button:focus {
              border-color: rgb(147 197 253 / var(--tw-border-opacity));
              outline: none;
              box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
              --tw-border-opacity: 1;
              --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) \
                var(--tw-ring-offset-color);
              --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
              --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
              --tw-ring-opacity: .5;
          }
          .footer {
              margin-bottom: 45px;
              margin-top: 35px;
              text-align: center;
              border-bottom: 1px solid #e5e5e5;
          }
          .footer>p {
              font-size: .8rem;
              display: inline-block;
              padding: 0 10px;
              transform: translateY(10px);
              background: white;
          }
          .dark .footer {
              border-color: #303030;
          }
          .dark .footer>p {
              background: #0b0f19;
          }
          .prompt h4{
              margin: 1.25em 0 .25em 0;
              font-weight: bold;
              font-size: 115%;
          }
  """

    block = gr.Blocks(css=css)

    with block:
        gr.HTML("""
              <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                <div
                  style="
                    display: inline-flex;
                    gap: 0.8rem;
                    font-size: 1.75rem;
                    justify-content: center;
                  "
                >
                <h1>
                FastDeploy Client
                </h1>
                </div>
                <p font-size: 94%">
                The client is used for creating requests to fastdeploy server.
                </p>
              </div>
          """)
        with gr.Group():
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        server_addr_text = gr.Textbox(
                            label="server ip",
                            show_label=True,
                            max_lines=1,
                            placeholder="localhost",
                            value=args.request_ip)
                        with gr.Column():
                            server_http_port_text = gr.Textbox(
                                label="server port",
                                show_label=True,
                                max_lines=1,
                                placeholder="8001",
                                value=args.request_port)
                            port_type_text = gr.Radio(
                                ['http', 'grpc'],
                                value=args.request_port_type,
                                label="port type")

                        server_metric_port_text = gr.Textbox(
                            label="metrics port",
                            show_label=True,
                            max_lines=1,
                            placeholder="8002",
                            value=args.request_metric)
                    with gr.Row():
                        model_name_text = gr.Textbox(
                            label="model name",
                            show_label=True,
                            max_lines=1,
                            placeholder="stable_diffusion",
                            value='stable_diffusion')
                        model_version_text = gr.Textbox(
                            label="model version",
                            show_label=True,
                            max_lines=1,
                            placeholder="1",
                            value="1")

            with gr.Box():
                input_column = gr.Column()
                with input_column:
                    gr.Markdown(
                        "Please input prompt text to describe contents")
                    input_text = gr.Textbox(label="Text", max_lines=1000)

                output_column = gr.Column(visible=False)
                with output_column:
                    gr.Markdown("Generated image")
                    output_image = gr.Image(interactive=False)

                with gr.Row():
                    component_submit_button = gr.Button("submit")
                    reset_button = gr.Button("reset")

            with gr.Box():
                with gr.Column():
                    gr.Markdown(
                        "Metrics（update automatically when submit request，or click update metrics button manually）"
                    )
                    output_html_table = gr.HTML(
                        label="metrics",
                        interactive=False,
                        show_label=False,
                        value=metrics_table_head_en.format('', ''))
                    update_metric_button = gr.Button("update metrics")

            status_text = gr.Textbox(
                label="status",
                show_label=True,
                max_lines=1,
                interactive=False)

        lang_text = gr.Textbox(
            label="lang",
            show_label=False,
            value='zh',
            max_lines=1,
            visible=False
        )  # This text box is only used for divide zh and en page

        # input_column
        # input_text
        # output_column
        # output_image
        # component_submit_button
        # reset_button

        def component_inference(*args):
            server_ip = args[0]
            port = args[1]
            port_type = args[2]
            metric_port = args[3]
            model_name = args[4]
            model_version = args[5]

            input_text_data = args[6]

            server_addr = server_ip + ':' + port
            if port_type == 'http':
                manager = _http_manager
            else:
                manager = _grpc_manager
            try:
                input_metas, output_metas = manager.get_model_meta(
                    server_addr, model_name, model_version)
            except Exception as e:
                return {status_text: str(e)}

            if server_ip and port and model_name and model_version:
                inputs = {}
                for i, input_meta in enumerate(input_metas):
                    if input_text_data:
                        print('input_text_data', input_text_data)
                        inputs[input_meta.name] = np.array(
                            [[input_text_data.encode('utf-8')]],
                            dtype=np.object_)
                    else:
                        return {
                            status_text:
                            "Please input prompt text before submit"
                        }
                try:
                    infer_results = manager.infer(server_addr, model_name,
                                                  model_version, inputs)
                    results = {status_text: 'Inference successfully'}
                    for i, (output_name,
                            data) in enumerate(infer_results.items()):
                        try:
                            results[output_image] = visualize_text_to_image(
                                data)
                        except Exception:
                            results[output_image] = None
                            return {
                                status_text: "Error to parse returned data"
                            }
                    if metric_port:
                        html_table = get_metric_data(server_ip, metric_port,
                                                     'zh')
                        results[output_html_table] = html_table
                    results[output_column] = gr.update(visible=True)
                    return results
                except Exception as e:
                    return {status_text: 'Error: {}'.format(e)}
            else:
                return {
                    status_text:
                    'Please input server ip, port first before submit'
                }

        def update_metric(server_ip, metrics_port, lang_text):
            if server_ip and metrics_port:
                try:
                    html_table = get_metric_data(server_ip, metrics_port, 'en')
                    return {
                        output_html_table: html_table,
                        status_text: "Update metrics successfully."
                    }
                except Exception as e:
                    return {status_text: 'Error: {}'.format(e)}
            else:
                return {
                    status_text: 'Please input server ip and metrics_port.'
                }

        def clear_contents(*args):
            results = {
                output_column: gr.update(visible=False),
                output_image: None,
                input_text: None
            }
            return results

        component_submit_button.click(
            fn=component_inference,
            inputs=[
                server_addr_text, server_http_port_text, port_type_text,
                server_metric_port_text, model_name_text, model_version_text,
                input_text
            ],
            outputs=[
                output_column, output_image, status_text, output_html_table
            ])

        reset_button.click(
            fn=clear_contents,
            inputs=[],
            outputs=[output_column, output_image, input_text])

        update_metric_button.click(
            fn=update_metric,
            inputs=[server_addr_text, server_metric_port_text, lang_text],
            outputs=[output_html_table, status_text])

        return block
