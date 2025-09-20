"""AI Model selection and parameter configuration dialog."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QItemSelection, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from novelwriter import CONFIG, SHARED
from novelwriter.ai import ModelInfo, NWAiApi, NWAiApiError

logger = logging.getLogger(__name__)


class ModelSelectorDialog(QDialog):
    """Dialog for selecting AI models and configuring parameters."""

    modelSelected = pyqtSignal(str, dict)  # model_id, model_metadata

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self.setObjectName("ModelSelectorDialog")
        self.setWindowTitle(self.tr("Select AI Model"))
        self.setModal(True)
        self.resize(800, 600)
        
        self._api: NWAiApi | None = None
        self._models: list[ModelInfo] = []
        self._selected_model: ModelInfo | None = None
        self._model_metadata: dict[str, Any] = {}
        
        self._setupUI()
        self._loadCurrentConfig()
        self._refreshModels()

    def _setupUI(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Create splitter for models list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: models list
        models_widget = self._createModelsWidget()
        splitter.addWidget(models_widget)
        
        # Right side: model details and parameters
        details_widget = self._createDetailsWidget()
        splitter.addWidget(details_widget)
        
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._acceptSelection)
        button_box.rejected.connect(self.reject)
        self._okButton = button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self._okButton:
            if self._okButton:
                self._okButton.setEnabled(False)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _createModelsWidget(self) -> QWidget:
        """Create the models list widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with refresh button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        models_label = QLabel(self.tr("Available Models"))
        self._refreshButton = QPushButton(self.tr("Refresh"))
        self._refreshButton.clicked.connect(self._refreshModels)
        header_layout.addWidget(models_label)
        header_layout.addStretch()
        header_layout.addWidget(self._refreshButton)
        layout.addLayout(header_layout)

        # Models tree view
        self._modelsTreeView = QTreeView()
        self._modelsTreeView.setObjectName("modelsTreeView")
        self._modelsTreeView.setRootIsDecorated(False)
        self._modelsTreeView.setAlternatingRowColors(True)
        self._modelsTreeView.setSelectionMode(QTreeView.SelectionMode.SingleSelection)

        self._modelsModel = QStandardItemModel()
        self._modelsModel.setHorizontalHeaderLabels([
            self.tr("Model"),
            self.tr("Owner"),
            self.tr("Input Limit"),
            self.tr("Output Limit")
        ])
        self._modelsTreeView.setModel(self._modelsModel)

        selection_model = self._modelsTreeView.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self._onModelSelectionChanged)

        layout.addWidget(self._modelsTreeView)
        widget.setLayout(layout)
        return widget

    def _createDetailsWidget(self) -> QWidget:
        """Create the model details and parameters widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Tab widget for details and parameters
        self._tabWidget = QTabWidget()
        
        # Model info tab
        info_tab = self._createModelInfoTab()
        self._tabWidget.addTab(info_tab, self.tr("Model Info"))
        
        # Parameters tab
        params_tab = self._createParametersTab()
        self._tabWidget.addTab(params_tab, self.tr("Parameters"))
        
        layout.addWidget(self._tabWidget)
        widget.setLayout(layout)
        return widget

    def _createModelInfoTab(self) -> QWidget:
        """Create the model information tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Model details
        self._modelDetailsBrowser = QTextBrowser()
        self._modelDetailsBrowser.setObjectName("modelDetailsBrowser")
        self._modelDetailsBrowser.setPlainText(self.tr("Select a model to view details"))
        layout.addWidget(self._modelDetailsBrowser)

        widget.setLayout(layout)
        return widget

    def _createParametersTab(self) -> QWidget:
        """Create the parameters configuration tab."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Generation parameters
        generation_group = QGroupBox(self.tr("Generation Parameters"))
        generation_layout = QFormLayout()

        # Temperature
        self._temperatureSpinBox = QDoubleSpinBox()
        self._temperatureSpinBox.setRange(0.0, 2.0)
        self._temperatureSpinBox.setSingleStep(0.1)
        self._temperatureSpinBox.setDecimals(2)
        self._temperatureSpinBox.setValue(0.7)
        generation_layout.addRow(self.tr("Temperature:"), self._temperatureSpinBox)

        # Max tokens
        self._maxTokensSpinBox = QSpinBox()
        self._maxTokensSpinBox.setRange(1, 32768)
        self._maxTokensSpinBox.setValue(2048)
        generation_layout.addRow(self.tr("Max Output Tokens:"), self._maxTokensSpinBox)

        generation_group.setLayout(generation_layout)
        layout.addWidget(generation_group)

        # Advanced parameters (placeholder for future expansion)
        advanced_group = QGroupBox(self.tr("Advanced"))
        advanced_layout = QFormLayout()
        
        # Placeholder for future parameters
        self._advancedPlaceholder = QLabel(self.tr("Additional parameters will be available in future versions."))
        self._advancedPlaceholder.setStyleSheet("color: gray; font-style: italic;")
        advanced_layout.addRow(self._advancedPlaceholder)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        layout.addStretch()
        widget.setLayout(layout)
        scroll_area.setWidget(widget)
        return scroll_area

    def _loadCurrentConfig(self) -> None:
        """Load current configuration values."""
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            return

        # Load temperature
        temperature = getattr(ai_config, "temperature", 0.7)
        self._temperatureSpinBox.setValue(float(temperature))

        # Load max tokens
        max_tokens = getattr(ai_config, "max_tokens", 2048)
        self._maxTokensSpinBox.setValue(int(max_tokens))

    def _refreshModels(self) -> None:
        """Refresh the models list from the API."""
        if not self._ensureApiAvailable():
            return

        self._refreshButton.setEnabled(False)
        self._refreshButton.setText(self.tr("Loading..."))
        
        try:
            if self._api:
                self._models = self._api.listAvailableModels(refresh=True)
            self._populateModelsTree()
            self._refreshButton.setText(self.tr("Refresh"))
        except NWAiApiError as exc:
            logger.error("Failed to refresh models: %s", exc)
            self._models = []
            self._populateModelsTree()
            self._modelDetailsBrowser.setPlainText(
                self.tr("Failed to load models: {0}").format(str(exc))
            )
            self._refreshButton.setText(self.tr("Retry"))
        finally:
            self._refreshButton.setEnabled(True)

    def _populateModelsTree(self) -> None:
        """Populate the models tree view."""
        self._modelsModel.clear()
        self._modelsModel.setHorizontalHeaderLabels([
            self.tr("Model"),
            self.tr("Owner"),
            self.tr("Input Limit"),
            self.tr("Output Limit")
        ])

        for model in self._models:
            name_item = QStandardItem(model.display_name)
            name_item.setData(model, Qt.ItemDataRole.UserRole)
            name_item.setToolTip(model.description or "")

            owner_item = QStandardItem(model.owned_by or "")
            input_item = QStandardItem(str(model.input_token_limit) if model.input_token_limit else "")
            output_item = QStandardItem(str(model.output_token_limit) if model.output_token_limit else "")

            self._modelsModel.appendRow([name_item, owner_item, input_item, output_item])

        # Auto-resize columns
        header = self._modelsTreeView.header()
        if header is not None:
            header.resizeSection(0, 250)  # Model name column wider
            header.setStretchLastSection(True)

    def _onModelSelectionChanged(
        self,
        selected: QItemSelection | None = None,
        deselected: QItemSelection | None = None,
    ) -> None:
        """Handle model selection change."""
        _ = selected, deselected
        selection_model = self._modelsTreeView.selectionModel()
        if selection_model is None:
            return
            
        selection = selection_model.selectedRows()
        if not selection:
            self._selected_model = None
            if self._okButton:
                self._okButton.setEnabled(False)
            self._modelDetailsBrowser.setPlainText(self.tr("Select a model to view details"))
            return

        index = selection[0]
        model = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(model, ModelInfo):
            return

        self._selected_model = model
        if self._okButton:
            self._okButton.setEnabled(True)
        self._updateModelDetails(model)

    def _updateModelDetails(self, model: ModelInfo) -> None:
        """Update the model details display."""
        details = []
        details.append(f"<h3>{model.display_name}</h3>")
        
        if model.description:
            details.append(f"<p><strong>{self.tr('Description')}:</strong> {model.description}</p>")
        
        details.append(f"<p><strong>{self.tr('Model ID')}:</strong> {model.id}</p>")
        
        if model.owned_by:
            details.append(f"<p><strong>{self.tr('Owner')}:</strong> {model.owned_by}</p>")
        
        if model.input_token_limit:
            details.append(f"<p><strong>{self.tr('Input Token Limit')}:</strong> {model.input_token_limit:,}</p>")
        
        if model.output_token_limit:
            details.append(f"<p><strong>{self.tr('Output Token Limit')}:</strong> {model.output_token_limit:,}</p>")
        
        if model.capabilities:
            details.append(f"<p><strong>{self.tr('Capabilities')}:</strong> {model.capabilities}</p>")

        self._modelDetailsBrowser.setHtml("".join(details))

    def _acceptSelection(self) -> None:
        """Accept the current selection and close dialog."""
        if self._selected_model is None:
            return

        # Collect parameters
        parameters = {
            "temperature": self._temperatureSpinBox.value(),
            "max_tokens": self._maxTokensSpinBox.value(),
        }

        # Emit signal with selected model and parameters
        self.modelSelected.emit(self._selected_model.id, {
            "model_metadata": self._selected_model.as_dict(),
            "parameters": parameters
        })
        
        self.accept()

    def _ensureApiAvailable(self) -> bool:
        """Ensure the AI API is available."""
        if self._api is not None:
            return True

        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            self._modelDetailsBrowser.setPlainText(
                self.tr("AI configuration not available.")
            )
            return False

        try:
            from novelwriter.ai import NWAiApi
            if SHARED.project:
                self._api = NWAiApi(SHARED.project)
            return True
        except Exception as exc:
            logger.error("Failed to initialize AI API: %s", exc)
            self._modelDetailsBrowser.setPlainText(
                self.tr("Failed to initialize AI API: {0}").format(str(exc))
            )
            return False


class ModelParametersDialog(QDialog):
    """Dialog for configuring model parameters only."""

    parametersChanged = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self.setObjectName("ModelParametersDialog")
        self.setWindowTitle(self.tr("Model Parameters"))
        self.setModal(True)
        self.resize(400, 300)
        
        self._setupUI()
        self._loadCurrentConfig()

    def _setupUI(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Parameters form
        form_layout = QFormLayout()

        # Temperature
        self._temperatureSpinBox = QDoubleSpinBox()
        self._temperatureSpinBox.setRange(0.0, 2.0)
        self._temperatureSpinBox.setSingleStep(0.1)
        self._temperatureSpinBox.setDecimals(2)
        self._temperatureSpinBox.setValue(0.7)
        form_layout.addRow(self.tr("Temperature:"), self._temperatureSpinBox)

        # Max tokens
        self._maxTokensSpinBox = QSpinBox()
        self._maxTokensSpinBox.setRange(1, 32768)
        self._maxTokensSpinBox.setValue(2048)
        form_layout.addRow(self.tr("Max Output Tokens:"), self._maxTokensSpinBox)

        layout.addLayout(form_layout)
        layout.addStretch()

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._acceptChanges)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _loadCurrentConfig(self) -> None:
        """Load current configuration values."""
        ai_config = getattr(CONFIG, "ai", None)
        if ai_config is None:
            return

        # Load temperature
        temperature = getattr(ai_config, "temperature", 0.7)
        self._temperatureSpinBox.setValue(float(temperature))

        # Load max tokens
        max_tokens = getattr(ai_config, "max_tokens", 2048)
        self._maxTokensSpinBox.setValue(int(max_tokens))

    def _acceptChanges(self) -> None:
        """Accept the parameter changes and close dialog."""
        parameters = {
            "temperature": self._temperatureSpinBox.value(),
            "max_tokens": self._maxTokensSpinBox.value(),
        }
        
        self.parametersChanged.emit(parameters)
        self.accept()