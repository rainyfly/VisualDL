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
import json

from .profiler_reader import ProfileReader
from visualdl.server.api import gen_result
from visualdl.server.api import result


class ProfilerApi(object):
    def __init__(self, logdir):
        self._reader = ProfileReader(logdir)

    @result()
    def runs(self):
        # print("I am here")
        return self._reader.runs()

    @result()
    def views(self, run):
        # print("I am here1")
        run_manager = self._reader.get_run_manager(run)
        if run_manager is None:
            return []
        # print(run_manager.get_views())
        return list(run_manager.get_views())

    @result()
    def workers(self, run, view):
        run_manager = self._reader.get_run_manager(run)
        return run_manager.get_workers(view)

    @result()
    def spans(self, run, worker):
        run_manager = self._reader.get_run_manager(run)
        return run_manager.get_spans(worker)

    @result()
    def timeunits(self):
        return ['ns', 'us', 'ms', 's']

    @result()
    def overview_environment(self, run, worker, span):
        run_manager = self._reader.get_run_manager(run)
        span = str(span)
        profiler_data = run_manager.get_profile_data(worker, span)
        result = profiler_data.get_device_infos()
        num_workers = len(run_manager.get_workers('Overview'))
        result['num_workers'] = num_workers
        return result

    @result()
    def model_perspective(self, run, worker, span, time_unit='ms'):
        run_manager = self._reader.get_run_manager(run)
        profiler_data = run_manager.get_profile_data(worker, span)
        return profiler_data.get_model_perspective(time_unit)

    @result()
    def model_perspective_perstep(self,
                                  run,
                                  worker,
                                  span,
                                  device_type,
                                  time_unit='ms'):
        run_manager = self._reader.get_run_manager(run)
        profiler_data = run_manager.get_profile_data(worker, span)
        return profiler_data.get_model_perspective_perstep(
            device_type, time_unit)

    @result()
    def event_type_perspective(self,
                               run,
                               worker,
                               span,
                               device_type,
                               time_unit='ms'):
        run_manager = self._reader.get_run_manager(run)
        profiler_data = run_manager.get_profile_data(worker, span)
        return profiler_data.get_event_type_perspective(device_type, time_unit)

    @result()
    def event_type_model_perspective(self, run, worker, span, time_unit='ms'):
        run_manager = self._reader.get_run_manager(run)
        profiler_data = run_manager.get_profile_data(worker, span)
        return profiler_data.get_event_type_model_perspective(time_unit)

    @result()
    def userdefined_perspective(self, run, worker, span, time_unit='ms'):
        run_manager = self._reader.get_run_manager(run)
        profiler_data = run_manager.get_profile_data(worker, span)
        return profiler_data.get_userdefined_perspective(time_unit)

    @result()
    def operator_pie(self, run, worker, span, topk, time_unit='ms'):
        pass

    @result()
    def operator_table(self,
                       run,
                       worker,
                       span,
                       group_by,
                       search_name,
                       time_unit='ms'):
        pass

    @result()
    def operator_stack_table(self,
                             run,
                             worker,
                             span,
                             op_name,
                             group_by,
                             input_shape,
                             time_unit='ms'):
        pass

    @result()
    def kernel_pie(self, run, worker, span, topk, time_unit='ms'):
        pass

    @result()
    def kernel_table(self,
                     run,
                     worker,
                     span,
                     group_by,
                     search_name,
                     time_unit='ms'):
        pass

    @result()
    def kernel_tc_pie(self, run, worker, span, topk, time_unit='ms'):
        pass

    @result()
    def distributed_info(self, run, worker, span):
        pass

    @result()
    def distributed_table(self, run, worker, span):
        pass

    @result()
    def trace(self, run, worker, span):
        pass

    @result()
    def memory_devices(self, run, worker, span):
        pass

    @result()
    def memory_curve(self, run, worker, span, device_type):
        pass

    @result()
    def memory_events(self, run, worker, span, device_type, min_size, max_size,
                      search_name):
        pass

    @result()
    def op_memory_events(self, run, worker, span, device_type, search_name):
        pass

    @result()
    def comparison_phase(self, base_run, base_worker, base_span, exp_run,
                         exp_worker, exp_span):
        pass

    @result()
    def comparison_phase_diff(self, base_run, base_worker, base_span, exp_run,
                              exp_worker, exp_span):
        pass

    @result()
    def comparison_phase_table(self, base_run, base_worker, base_span, exp_run,
                               exp_worker, exp_span):
        pass

    @result()
    def comparison_phase_inner(self, base_run, base_worker, base_span, exp_run,
                               exp_worker, exp_span, phase_name):
        pass

    @result()
    def comparison_phase_diff_inner(self, base_run, base_worker, base_span,
                                    exp_run, exp_worker, exp_span, phase_name):
        pass

    @result()
    def comparison_phase_table_inner(self, base_run, base_worker, base_span,
                                     exp_run, exp_worker, exp_span,
                                     phase_name):
        pass


def create_profiler_api_call(logdir):
    api = ProfilerApi(logdir)
    routes = {
        'runs': (api.runs, []),
        'views': (api.views, ["run"]),
        'workers': (api.workers, ["run", "view"]),
        'spans': (api.spans, ["run", "worker"]),
        'timeunits': (api.timeunits, []),
        'overview/environment': (api.overview_environment,
                                 ["run", "worker", "span"]),
        'overview/model_perspective': (api.model_perspective,
                                       ["run", "worker", "span", "time_unit"]),
        'overview/model_perspective_perstep': (api.model_perspective_perstep, [
            "run", "worker", "span", "device_type", "time_unit"
        ]),
        'overview/event_type_perspective': (api.event_type_perspective, [
            "run", "worker", "span", "device_type", "time_unit"
        ]),
        'overview/event_type_model_perspective':
        (api.event_type_model_perspective,
         ["run", "worker", "span", "time_unit"]),
        'overview/userdefined_perspective':
        (api.userdefined_perspective, ["run", "worker", "span", "time_unit"]),
        'operator/pie': (api.operator_pie,
                         ["run", "worker", "span", "time_unit", "topk"]),
        'operator/table': (api.operator_table, [
            "run", "worker", "span", "time_unit", "group_by", "search_name"
        ]),
        'operator/stack_table': (api.operator_stack_table, [
            "run", "worker", "span", "time_unit", "op_name", "group_by",
            "input_shape"
        ]),
        'kernel/pie': (api.kernel_pie,
                       ["run", "worker", "span", "time_unit", "topk"]),
        'kernel/tensorcore_pie':
        (api.kernel_tc_pie, ["run", "worker", "span", "time_unit", "topk"]),
        'kernel/table': (api.kernel_table, [
            "run", "worker", "span", "time_unit", "group_by", "search_name"
        ]),
        'distributed/info': (api.distributed_info, ["run", "worker", "span"]),
        'distributed/table': (api.distributed_table, ["run", "worker",
                                                      "span"]),
        'trace': (api.trace, ["run", "worker", "span"]),
        'memory/devices': (api.memory_devices, ["run", "worker", "span"]),
        'memory/curve': (api.memory_curve,
                         ["run", "worker", "span", "device_type"]),
        'memory/memory_events': (api.memory_events, [
            "run", "worker", "span", "device_type", "min_size", "max_size",
            "search_name"
        ]),
        'memory/op_memory_events': (api.op_memory_events, [
            "run", "worker", "span", "device_type", "search_name"
        ]),
        'comparison/phase': (api.comparison_phase, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span"
        ]),
        'comparison/phase_diff': (api.comparison_phase_diff, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span"
        ]),
        'comparison/phase_table': (api.comparison_phase_table, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span"
        ]),
        'comparison/phase_inner': (api.comparison_phase_inner, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span", "phase_name"
        ]),
        'comparison/phase_diff_inner': (api.comparison_phase_diff_inner, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span", "phase_name"
        ]),
        'comparison/phase_table_inner': (api.comparison_phase_table_inner, [
            "base_run", "base_worker", "base_span", "exp_run", "exp_worker",
            "exp_span", "phase_name"
        ])
    }

    def call(path: str, args):
        route = routes.get(path)
        if not route:
            return json.dumps(gen_result(
                status=1, msg='api not found')), 'application/json', None
        method, call_arg_names = route
        call_args = [args.get(name) for name in call_arg_names]
        # print(method, call_arg_names)
        return method(*call_args)

    return call
