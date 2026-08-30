"""
bridge.py -- QWebChannel bridge exposed to JavaScript.

Every method here should be a thin wrapper: parse/serialize data and
delegate to experiment_manager / app_state. No business logic here.
"""

import json

from PySide6.QtCore import QObject, Slot, Signal

from . import experiment_manager
from .app_state import AppState


class Bridge(QObject):
    stateChanged = Signal(str)

    def __init__(self, app_state: AppState):
        super().__init__()
        self.state = app_state

    @Slot(result=str)
    def getExperiments(self):
        experiments = experiment_manager.get_experiments()
        return json.dumps(experiments)

    @Slot(str, result=str)
    def getExperiment(self, experiment_id):
        exp = experiment_manager.get_experiment(experiment_id)
        return json.dumps(exp) if exp else "null"

    @Slot(str, result=str)
    def createExperiment(self, experiment_json):
        try:
            payload = json.loads(experiment_json)
            new_exp = experiment_manager.create_experiment(
                name=payload.get("name", ""),
                description=payload.get("description", ""),
                steps=payload.get("steps", []),
            )
            return json.dumps({"success": True, "experiment": new_exp})
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)})

    @Slot(str, str, result=str)
    def saveExperiment(self, experiment_id, experiment_json):
        payload = json.loads(experiment_json)
        updated = experiment_manager.update_experiment(
            experiment_id,
            name=payload.get("name"),
            description=payload.get("description"),
            steps=payload.get("steps"),
        )
        if updated:
            return json.dumps({"success": True, "experiment": updated})
        return json.dumps({"success": False, "error": "Experiment not found"})

    @Slot(str, result=bool)
    def deleteExperiment(self, experiment_id):
        return experiment_manager.delete_experiment(experiment_id)

    @Slot(str, result=str)
    def loadExperiment(self, experiment_id):
        exp = experiment_manager.get_experiment(experiment_id)
        if not exp:
            return json.dumps({"success": False, "error": "Not found"})
        self.state.set_experiment(experiment_id)
        self._emit_state()
        return json.dumps({"success": True, "experiment": exp})

    @Slot(str)
    def selectVideoSource(self, source):
        self.state.set_video_source(source)
        self._emit_state()

    @Slot(result=str)
    def getState(self):
        return json.dumps(self.state.to_dict())

    @Slot()
    def startMonitoring(self):
        self.state.start_monitoring()
        self._emit_state()

    @Slot()
    def pauseMonitoring(self):
        self.state.pause_monitoring()
        self._emit_state()

    @Slot()
    def stopMonitoring(self):
        self.state.stop_monitoring()
        self._emit_state()

    def _emit_state(self):
        self.stateChanged.emit(json.dumps(self.state.to_dict()))