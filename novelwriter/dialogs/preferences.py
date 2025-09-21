"""
novelWriter – GUI Preferences
=============================

File History:
Created:   2019-06-10 [0.1.5] GuiPreferences
Rewritten: 2024-01-08 [2.3b1] GuiPreferences

This file is a part of novelWriter
Copyright (C) 2019 Veronica Berglyd Olsen and novelWriter contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""  # noqa
from __future__ import annotations

import importlib.util
import logging

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCloseEvent, QKeyEvent, QKeySequence
from PyQt6.QtWidgets import (
    QCompleter, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMenu,
    QPushButton, QVBoxLayout, QWidget
)

from novelwriter import CONFIG, SHARED
from novelwriter.common import compact, describeFont, processDialogSymbols, uniqueCompact
from novelwriter.config import DEF_GUI_DARK, DEF_GUI_LIGHT, DEF_ICONS, DEF_TREECOL
from novelwriter.constants import nwLabels, nwQuotes, nwUnicode, trConst
from novelwriter.dialogs.quotes import GuiQuoteSelect
from novelwriter.extensions.configlayout import NColorLabel, NScrollableForm
from novelwriter.extensions.modified import (
    NComboBox, NDialog, NDoubleSpinBox, NIconToolButton, NSpinBox
)
from novelwriter.extensions.pagedsidebar import NPagedSideBar
from novelwriter.extensions.switch import NSwitch
from novelwriter.types import QtAlignCenter, QtDialogCancel, QtDialogSave

logger = logging.getLogger(__name__)


class GuiPreferences(NDialog):
    """GUI: Preferences Dialog."""

    newPreferencesReady = pyqtSignal(bool, bool, bool, bool)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent=parent)

        logger.debug("Create: GuiPreferences")
        self.setObjectName("GuiPreferences")
        self.setWindowTitle(self.tr("Preferences"))
        self.setMinimumSize(600, 500)
        self.resize(*CONFIG.prefsWinSize)

        # Title
        self.titleLabel = NColorLabel(
            self.tr("Preferences"), self, color=SHARED.theme.helpText,
            scale=NColorLabel.HEADER_SCALE, indent=4,
        )

        # Search Box
        self.searchAction = QAction(SHARED.theme.getIcon("search"), "")
        self.searchAction.triggered.connect(self._gotoSearch)

        self.searchText = QLineEdit(self)
        self.searchText.setPlaceholderText(self.tr("Search"))
        self.searchText.setMinimumWidth(200)
        self.searchText.addAction(self.searchAction, QLineEdit.ActionPosition.TrailingPosition)

        # SideBar
        self.sidebar = NPagedSideBar(self)
        self.sidebar.setLabelColor(SHARED.theme.helpText)
        self.sidebar.setAccessibleName(self.titleLabel.text())
        self.sidebar.buttonClicked.connect(self._sidebarClicked)

        # Form
        self.mainForm = NScrollableForm(self)
        self.mainForm.setHelpTextStyle(SHARED.theme.helpText)

        # Buttons
        self.buttonBox = QDialogButtonBox(QtDialogSave | QtDialogCancel, self)
        self.buttonBox.accepted.connect(self._doSave)
        self.buttonBox.rejected.connect(self.reject)

        # Assemble
        self.searchBox = QHBoxLayout()
        self.searchBox.addWidget(self.titleLabel)
        self.searchBox.addStretch(1)
        self.searchBox.addWidget(self.searchText, 1)

        self.mainBox = QHBoxLayout()
        self.mainBox.addWidget(self.sidebar)
        self.mainBox.addWidget(self.mainForm)
        self.mainBox.setContentsMargins(0, 0, 0, 0)

        self.outerBox = QVBoxLayout()
        self.outerBox.addLayout(self.searchBox)
        self.outerBox.addLayout(self.mainBox)
        self.outerBox.addWidget(self.buttonBox)
        self.outerBox.setSpacing(8)

        self.setLayout(self.outerBox)
        self.setSizeGripEnabled(True)

        # Build Form
        self.buildForm()

        # Populate Search
        self.searchCompleter = QCompleter(self.mainForm.labels, self)
        self.searchCompleter.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.searchCompleter.setFilterMode(Qt.MatchFlag.MatchContains)
        self.searchCompleter.activated.connect(self._gotoSearch)

        self.searchText.setCompleter(self.searchCompleter)

        logger.debug("Ready: GuiPreferences")

    def __del__(self) -> None:  # pragma: no cover
        logger.debug("Delete: GuiPreferences")

    def buildForm(self) -> None:
        """Build the settings form."""
        section = 0
        iSz = SHARED.theme.baseIconSize
        boxFixed = 6*SHARED.theme.textNWidth

        # Temporary Variables
        self._guiFont = CONFIG.guiFont
        self._textFont = CONFIG.textFont

        # Label
        self.sidebar.addLabel(self.tr("General"))

        # Appearance
        # ==========

        title = self.tr("Appearance")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Display Language
        self.guiLocale = NComboBox(self)
        self.guiLocale.setMinimumWidth(200)
        for lang, name in CONFIG.listLanguages(CONFIG.LANG_NW):
            self.guiLocale.addItem(name, lang)
        self.guiLocale.setCurrentData(CONFIG.guiLocale, "en_GB")

        self.mainForm.addRow(
            self.tr("Display language"), self.guiLocale,
            self.tr("Requires restart to take effect."), stretch=(3, 2)
        )

        # Colour Theme
        self.lightTheme = NComboBox(self)
        self.lightTheme.setMinimumWidth(200)
        self.darkTheme = NComboBox(self)
        self.darkTheme.setMinimumWidth(200)
        for key, theme in SHARED.theme.colourThemes.items():
            if theme.dark:
                self.darkTheme.addItem(theme.name, key)
            else:
                self.lightTheme.addItem(theme.name, key)

        self.lightTheme.setCurrentData(CONFIG.lightTheme, DEF_GUI_LIGHT)
        self.darkTheme.setCurrentData(CONFIG.darkTheme, DEF_GUI_DARK)

        self.mainForm.addRow(
            self.tr("Light colour theme"), self.lightTheme,
            self.tr("You can change theme mode from the sidebar."), stretch=(3, 2)
        )
        self.mainForm.addRow(
            self.tr("Dark colour theme"), self.darkTheme,
            self.tr("You can change theme mode from the sidebar."), stretch=(3, 2)
        )

        # Icon Theme
        self.iconTheme = NComboBox(self)
        self.iconTheme.setMinimumWidth(200)
        for key, theme in SHARED.theme.iconCache.iconThemes.items():
            self.iconTheme.addItem(theme.name, key)

        self.iconTheme.setCurrentData(CONFIG.iconTheme, DEF_ICONS)

        self.mainForm.addRow(
            self.tr("Icon theme"), self.iconTheme,
            self.tr("User interface icon theme."), stretch=(3, 2)
        )

        # Application Font Family
        self.guiFont = QLineEdit(self)
        self.guiFont.setReadOnly(True)
        self.guiFont.setMinimumWidth(162)
        self.guiFont.setText(describeFont(self._guiFont))
        self.guiFont.setCursorPosition(0)
        self.guiFontButton = NIconToolButton(self, iSz, "font")
        self.guiFontButton.clicked.connect(self._selectGuiFont)
        self.mainForm.addRow(
            self.tr("Application font"), self.guiFont,
            self.tr("Requires restart to take effect."), stretch=(3, 2),
            button=self.guiFontButton
        )

        # Vertical Scrollbars
        self.hideVScroll = NSwitch(self)
        self.hideVScroll.setChecked(CONFIG.hideVScroll)
        self.mainForm.addRow(
            self.tr("Hide vertical scroll bars in main windows"), self.hideVScroll,
            self.tr("Scrolling available with mouse wheel and keys only.")
        )

        # Horizontal Scrollbars
        self.hideHScroll = NSwitch(self)
        self.hideHScroll.setChecked(CONFIG.hideHScroll)
        self.mainForm.addRow(
            self.tr("Hide horizontal scroll bars in main windows"), self.hideHScroll,
            self.tr("Scrolling available with mouse wheel and keys only.")
        )

        # Native Font Dialog
        self.nativeFont = NSwitch(self)
        self.nativeFont.setChecked(CONFIG.nativeFont)
        self.mainForm.addRow(
            self.tr("Use the system's font selection dialog"), self.nativeFont,
            self.tr("Turn off to use the Qt font dialog, which may have more options.")
        )

        # Use Character Count
        self.useCharCount = NSwitch(self)
        self.useCharCount.setChecked(CONFIG.useCharCount)
        self.mainForm.addRow(
            self.tr("Prefer character count over word count"), self.useCharCount,
            self.tr("Display character count instead where available.")
        )

        # Document Style
        # ==============

        title = self.tr("Document Style")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Document Font Family
        self.textFont = QLineEdit(self)
        self.textFont.setReadOnly(True)
        self.textFont.setMinimumWidth(162)
        self.textFont.setText(describeFont(CONFIG.textFont))
        self.textFont.setCursorPosition(0)
        self.textFontButton = NIconToolButton(self, iSz, "font")
        self.textFontButton.clicked.connect(self._selectTextFont)
        self.mainForm.addRow(
            self.tr("Document font"), self.textFont,
            self.tr("Applies to both document editor and viewer."), stretch=(3, 2),
            button=self.textFontButton
        )

        # Document Path
        self.showFullPath = NSwitch(self)
        self.showFullPath.setChecked(CONFIG.showFullPath)
        self.mainForm.addRow(
            self.tr("Show full path in document header"), self.showFullPath,
            self.tr("Add the parent folder names to the header.")
        )

        # Include Notes in Word Count
        self.incNotesWCount = NSwitch(self)
        self.incNotesWCount.setChecked(CONFIG.incNotesWCount)
        self.mainForm.addRow(
            self.tr("Include project notes in status bar word count"), self.incNotesWCount
        )

        # Project View
        # ============

        title = self.tr("Project View")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Tree Icon Colours
        self.iconColTree = NComboBox(self)
        self.iconColTree.setMinimumWidth(200)
        self.iconColTree.addItem(self.tr("Theme Colours"), DEF_TREECOL)
        for key, label in nwLabels.THEME_COLORS.items():
            self.iconColTree.addItem(trConst(label), key)
        self.iconColTree.setCurrentData(CONFIG.iconColTree, DEF_TREECOL)

        self.mainForm.addRow(
            self.tr("Project tree icon colours"), self.iconColTree,
            self.tr("Override colours for project icons."), stretch=(3, 2)
        )

        # Keep Theme Colours on Documents
        self.iconColDocs = NSwitch(self)
        self.iconColDocs.setChecked(CONFIG.iconColDocs)
        self.mainForm.addRow(
            self.tr("Keep theme colours on documents"), self.iconColDocs,
            self.tr("Only override icon colours for folders.")
        )

        # Emphasise Labels
        self.emphLabels = NSwitch(self)
        self.emphLabels.setChecked(CONFIG.emphLabels)
        self.mainForm.addRow(
            self.tr("Emphasise partition and chapter labels"), self.emphLabels,
            self.tr("Makes them stand out in the project tree."),
        )

        # Behaviour
        # =========

        title = self.tr("Behaviour")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Document Save Timer
        self.autoSaveDoc = NSpinBox(self)
        self.autoSaveDoc.setMinimum(5)
        self.autoSaveDoc.setMaximum(600)
        self.autoSaveDoc.setSingleStep(1)
        self.autoSaveDoc.setValue(CONFIG.autoSaveDoc)
        self.mainForm.addRow(
            self.tr("Save document interval"), self.autoSaveDoc,
            self.tr("How often the document is automatically saved."), unit=self.tr("seconds")
        )

        # Project Save Timer
        self.autoSaveProj = NSpinBox(self)
        self.autoSaveProj.setMinimum(5)
        self.autoSaveProj.setMaximum(600)
        self.autoSaveProj.setSingleStep(1)
        self.autoSaveProj.setValue(CONFIG.autoSaveProj)
        self.mainForm.addRow(
            self.tr("Save project interval"), self.autoSaveProj,
            self.tr("How often the project is automatically saved."), unit=self.tr("seconds")
        )

        # Ask before exiting novelWriter
        self.askBeforeExit = NSwitch(self)
        self.askBeforeExit.setChecked(CONFIG.askBeforeExit)
        self.mainForm.addRow(
            self.tr("Ask before exiting novelWriter"), self.askBeforeExit,
            self.tr("Only applies when a project is open.")
        )

        # Project Backup
        # ==============

        title = self.tr("Project Backup")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Backup Path
        self.backupPath = CONFIG.backupPath()
        self.backupGetPath = QPushButton(SHARED.theme.getIcon("browse"), self.tr("Browse"), self)
        self.backupGetPath.setIconSize(iSz)
        self.backupGetPath.clicked.connect(self._backupFolder)
        self.mainForm.addRow(
            self.tr("Backup storage location"), self.backupGetPath,
            self.tr("Path: {0}").format(self.backupPath), editable="backupPath"
        )

        # Run When Closing
        self.backupOnClose = NSwitch(self)
        self.backupOnClose.setChecked(CONFIG.backupOnClose)
        self.backupOnClose.toggled.connect(self._toggledBackupOnClose)
        self.mainForm.addRow(
            self.tr("Run backup when the project is closed"), self.backupOnClose,
            self.tr("Can be overridden for individual projects in Project Settings.")
        )

        # Ask Before Backup
        # Only enabled when "Run when closing" is checked
        self.askBeforeBackup = NSwitch(self)
        self.askBeforeBackup.setChecked(CONFIG.askBeforeBackup)
        self.askBeforeBackup.setEnabled(CONFIG.backupOnClose)
        self.mainForm.addRow(
            self.tr("Ask before running backup"), self.askBeforeBackup,
            self.tr("If off, backups will run in the background.")
        )

        # Session Timer
        # =============

        title = self.tr("Session Timer")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Pause When Idle
        self.stopWhenIdle = NSwitch(self)
        self.stopWhenIdle.setChecked(CONFIG.stopWhenIdle)
        self.mainForm.addRow(
            self.tr("Pause the session timer when not writing"), self.stopWhenIdle,
            self.tr("Also pauses when the application window does not have focus.")
        )

        # Inactive Time for Idle
        self.userIdleTime = NDoubleSpinBox(self)
        self.userIdleTime.setMinimum(0.5)
        self.userIdleTime.setMaximum(600.0)
        self.userIdleTime.setSingleStep(0.5)
        self.userIdleTime.setDecimals(1)
        self.userIdleTime.setValue(CONFIG.userIdleTime/60.0)
        self.mainForm.addRow(
            self.tr("Editor inactive time before pausing timer"), self.userIdleTime,
            self.tr("User activity includes typing and changing the content."),
            unit=self.tr("minutes")
        )

        # Label
        self.sidebar.addLabel(self.tr("Writing"))

        # Text Flow
        # =========

        title = self.tr("Text Flow")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Max Text Width in Normal Mode
        self.textWidth = NSpinBox(self)
        self.textWidth.setMinimum(0)
        self.textWidth.setMaximum(10000)
        self.textWidth.setSingleStep(10)
        self.textWidth.setValue(CONFIG.textWidth)
        self.mainForm.addRow(
            self.tr('Maximum text width in "Normal Mode"'), self.textWidth,
            self.tr("Set to 0 to disable this feature."), unit=self.tr("px")
        )

        # Max Text Width in Focus Mode
        self.focusWidth = NSpinBox(self)
        self.focusWidth.setMinimum(200)
        self.focusWidth.setMaximum(10000)
        self.focusWidth.setSingleStep(10)
        self.focusWidth.setValue(CONFIG.focusWidth)
        self.mainForm.addRow(
            self.tr('Maximum text width in "Focus Mode"'), self.focusWidth,
            self.tr("The maximum width cannot be disabled."), unit=self.tr("px")
        )

        # Focus Mode Footer
        self.hideFocusFooter = NSwitch(self)
        self.hideFocusFooter.setChecked(CONFIG.hideFocusFooter)
        self.mainForm.addRow(
            self.tr('Hide document footer in "Focus Mode"'), self.hideFocusFooter,
            self.tr("Hide the information bar in the document editor.")
        )

        # Justify Text
        self.doJustify = NSwitch(self)
        self.doJustify.setChecked(CONFIG.doJustify)
        self.mainForm.addRow(
            self.tr("Justify the text margins"), self.doJustify,
            self.tr("Applies to both document editor and viewer."),
        )

        # Document Margins
        self.textMargin = NSpinBox(self)
        self.textMargin.setMinimum(0)
        self.textMargin.setMaximum(900)
        self.textMargin.setSingleStep(1)
        self.textMargin.setValue(CONFIG.textMargin)
        self.mainForm.addRow(
            self.tr("Minimum text margin"), self.textMargin,
            self.tr("Applies to both document editor and viewer."),
            unit=self.tr("px")
        )

        # Tab Width
        self.tabWidth = NSpinBox(self)
        self.tabWidth.setMinimum(0)
        self.tabWidth.setMaximum(200)
        self.tabWidth.setSingleStep(1)
        self.tabWidth.setValue(CONFIG.tabWidth)
        self.mainForm.addRow(
            self.tr("Tab width"), self.tabWidth,
            self.tr("The width of a tab key press in the editor and viewer."),
            unit=self.tr("px")
        )

        # Text Editing
        # ============

        title = self.tr("Text Editing")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Spell Checking
        self.spellLanguage = NComboBox(self)
        self.spellLanguage.setMinimumWidth(200)

        if CONFIG.hasEnchant:
            for tag, language in SHARED.spelling.listDictionaries():
                self.spellLanguage.addItem(language, tag)
        else:
            self.spellLanguage.addItem(nwUnicode.U_EMDASH, "")
            self.spellLanguage.setEnabled(False)

        if (idx := self.spellLanguage.findData(CONFIG.spellLanguage)) != -1:
            self.spellLanguage.setCurrentIndex(idx)

        self.mainForm.addRow(
            self.tr("Spell check language"), self.spellLanguage,
            self.tr("Available languages are determined by your system."), stretch=(3, 2)
        )

        # Auto-Select Word Under Cursor
        self.autoSelect = NSwitch(self)
        self.autoSelect.setChecked(CONFIG.autoSelect)
        self.mainForm.addRow(
            self.tr("Auto-select word under cursor"), self.autoSelect,
            self.tr("Apply formatting to word under cursor if no selection is made.")
        )

        # Cursor Width
        self.cursorWidth = NSpinBox(self)
        self.cursorWidth.setMinimum(1)
        self.cursorWidth.setMaximum(20)
        self.cursorWidth.setSingleStep(1)
        self.cursorWidth.setValue(CONFIG.cursorWidth)
        self.mainForm.addRow(
            self.tr("Cursor width"), self.cursorWidth,
            self.tr("The width of the text cursor of the editor."),
            unit=self.tr("px")
        )

        # Highlight Current Line
        self.lineHighlight = NSwitch(self)
        self.lineHighlight.setChecked(CONFIG.lineHighlight)
        self.mainForm.addRow(
            self.tr("Highlight current line"), self.lineHighlight
        )

        # Show Tabs and Spaces
        self.showTabsNSpaces = NSwitch(self)
        self.showTabsNSpaces.setChecked(CONFIG.showTabsNSpaces)
        self.mainForm.addRow(
            self.tr("Show tabs and spaces"), self.showTabsNSpaces
        )

        # Show Line Endings
        self.showLineEndings = NSwitch(self)
        self.showLineEndings.setChecked(CONFIG.showLineEndings)
        self.mainForm.addRow(
            self.tr("Show line endings"), self.showLineEndings
        )

        # Editor Scrolling
        # ================

        title = self.tr("Editor Scrolling")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Scroll Past End
        self.scrollPastEnd = NSwitch(self)
        self.scrollPastEnd.setChecked(CONFIG.scrollPastEnd)
        self.mainForm.addRow(
            self.tr("Scroll past the end of the document"), self.scrollPastEnd,
            self.tr("Also centres the cursor when scrolling.")
        )

        # Typewriter Scrolling
        self.autoScroll = NSwitch(self)
        self.autoScroll.setChecked(CONFIG.autoScroll)
        self.mainForm.addRow(
            self.tr("Typewriter style scrolling when you type"), self.autoScroll,
            self.tr("Keeps the cursor at a fixed vertical position.")
        )

        # Typewriter Position
        self.autoScrollPos = NSpinBox(self)
        self.autoScrollPos.setMinimum(10)
        self.autoScrollPos.setMaximum(90)
        self.autoScrollPos.setSingleStep(1)
        self.autoScrollPos.setValue(int(CONFIG.autoScrollPos))
        self.mainForm.addRow(
            self.tr("Minimum position for Typewriter scrolling"), self.autoScrollPos,
            self.tr("Percentage of the editor height from the top."), unit="%"
        )

        # Text Highlighting
        # =================

        title = self.tr("Text Highlighting")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Dialogue Quotes
        self.dialogStyle = NComboBox(self)
        self.dialogStyle.addItem(self.tr("None"), 0)
        self.dialogStyle.addItem(self.tr("Single Quotes"), 1)
        self.dialogStyle.addItem(self.tr("Double Quotes"), 2)
        self.dialogStyle.addItem(self.tr("Both"), 3)
        self.dialogStyle.setCurrentData(CONFIG.dialogStyle, 2)
        self.mainForm.addRow(
            self.tr("Highlight dialogue"), self.dialogStyle,
            self.tr("Applies to the selected quote styles.")
        )

        # Open-Ended Dialogue
        self.allowOpenDial = NSwitch(self)
        self.allowOpenDial.setChecked(CONFIG.allowOpenDial)
        self.mainForm.addRow(
            self.tr("Allow open-ended dialogue"), self.allowOpenDial,
            self.tr("Highlight dialogue line with no closing quote.")
        )

        # Alternative Dialogue
        self.altDialogOpen = QLineEdit(self)
        self.altDialogOpen.setMaxLength(4)
        self.altDialogOpen.setFixedWidth(boxFixed)
        self.altDialogOpen.setAlignment(QtAlignCenter)
        self.altDialogOpen.setText(CONFIG.altDialogOpen)

        self.altDialogClose = QLineEdit(self)
        self.altDialogClose.setMaxLength(4)
        self.altDialogClose.setFixedWidth(boxFixed)
        self.altDialogClose.setAlignment(QtAlignCenter)
        self.altDialogClose.setText(CONFIG.altDialogClose)

        self.mainForm.addRow(
            self.tr("Alternative dialogue symbols"), [self.altDialogOpen, self.altDialogClose],
            self.tr("Custom highlighting of dialogue text.")
        )

        # Dialogue Line
        self.mnLineSymbols = QMenu(self)
        for symbol in nwQuotes.ALLOWED:
            label = trConst(nwQuotes.SYMBOLS.get(symbol, nwQuotes.DASHES.get(symbol, "None")))
            self.mnLineSymbols.addAction(
                f"[ {symbol } ] {label}",
                lambda symbol=symbol: self._insertDialogLineSymbol(symbol)
            )

        self.dialogLine = QLineEdit(self)
        self.dialogLine.setMinimumWidth(100)
        self.dialogLine.setAlignment(QtAlignCenter)
        self.dialogLine.setText(" ".join(CONFIG.dialogLine))

        self.dialogLineButton = NIconToolButton(self, iSz, "add", "green")
        self.dialogLineButton.setMenu(self.mnLineSymbols)

        self.mainForm.addRow(
            self.tr("Dialogue line symbols"), self.dialogLine,
            self.tr("Lines starting with any of these symbols are dialogue."),
            button=self.dialogLineButton
        )

        # Narrator Break
        self.narratorBreak = NComboBox(self)
        self.narratorDialog = NComboBox(self)
        for key, value in nwQuotes.DASHES.items():
            label = trConst(value)
            self.narratorBreak.addItem(label, key)
            self.narratorDialog.addItem(label, key)

        self.narratorBreak.setCurrentData(CONFIG.narratorBreak, "")
        self.narratorDialog.setCurrentData(CONFIG.narratorDialog, "")

        self.mainForm.addRow(
            self.tr("Narrator break symbol"), self.narratorBreak,
            self.tr("Symbol to indicate a narrator break in dialogue.")
        )
        self.mainForm.addRow(
            self.tr("Alternating dialogue/narration symbol"), self.narratorDialog,
            self.tr("Alternates dialogue highlighting within any paragraph.")
        )

        # Emphasis
        self.highlightEmph = NSwitch(self)
        self.highlightEmph.setChecked(CONFIG.highlightEmph)
        self.mainForm.addRow(
            self.tr("Add highlight colour to emphasised text"), self.highlightEmph,
            self.tr("Applies to the document editor only.")
        )

        # Additional Spaces
        self.showMultiSpaces = NSwitch(self)
        self.showMultiSpaces.setChecked(CONFIG.showMultiSpaces)
        self.mainForm.addRow(
            self.tr("Highlight multiple spaces between words"), self.showMultiSpaces,
            self.tr("Applies to the document editor only.")
        )

        # Text Automation
        # ===============

        title = self.tr("Text Automation")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Auto-Replace as You Type Main Switch
        self.doReplace = NSwitch(self)
        self.doReplace.setChecked(CONFIG.doReplace)
        self.doReplace.toggled.connect(self._toggleAutoReplaceMain)
        self.mainForm.addRow(
            self.tr("Auto-replace text as you type"), self.doReplace,
            self.tr("Allow the editor to replace symbols as you type.")
        )

        # Auto-Replace Single Quotes
        self.doReplaceSQuote = NSwitch(self)
        self.doReplaceSQuote.setChecked(CONFIG.doReplaceSQuote)
        self.doReplaceSQuote.setEnabled(CONFIG.doReplace)
        self.mainForm.addRow(
            self.tr("Auto-replace single quotes"), self.doReplaceSQuote,
            self.tr("Try to guess which is an opening or a closing quote.")
        )

        # Auto-Replace Double Quotes
        self.doReplaceDQuote = NSwitch(self)
        self.doReplaceDQuote.setChecked(CONFIG.doReplaceDQuote)
        self.doReplaceDQuote.setEnabled(CONFIG.doReplace)
        self.mainForm.addRow(
            self.tr("Auto-replace double quotes"), self.doReplaceDQuote,
            self.tr("Try to guess which is an opening or a closing quote.")
        )

        # Auto-Replace Hyphens
        self.doReplaceDash = NSwitch(self)
        self.doReplaceDash.setChecked(CONFIG.doReplaceDash)
        self.doReplaceDash.setEnabled(CONFIG.doReplace)
        self.mainForm.addRow(
            self.tr("Auto-replace dashes"), self.doReplaceDash,
            self.tr("Double and triple hyphens become short and long dashes.")
        )

        # Auto-Replace Dots
        self.doReplaceDots = NSwitch(self)
        self.doReplaceDots.setChecked(CONFIG.doReplaceDots)
        self.doReplaceDots.setEnabled(CONFIG.doReplace)
        self.mainForm.addRow(
            self.tr("Auto-replace dots"), self.doReplaceDots,
            self.tr("Three consecutive dots become ellipsis.")
        )

        # Pad Before
        self.fmtPadBefore = QLineEdit(self)
        self.fmtPadBefore.setMaxLength(32)
        self.fmtPadBefore.setMinimumWidth(150)
        self.fmtPadBefore.setText(CONFIG.fmtPadBefore)
        self.mainForm.addRow(
            self.tr("Insert non-breaking space before"), self.fmtPadBefore,
            self.tr("Automatically add space before any of these symbols."), stretch=(2, 1)
        )

        # Pad After
        self.fmtPadAfter = QLineEdit(self)
        self.fmtPadAfter.setMaxLength(32)
        self.fmtPadAfter.setMinimumWidth(150)
        self.fmtPadAfter.setText(CONFIG.fmtPadAfter)
        self.mainForm.addRow(
            self.tr("Insert non-breaking space after"), self.fmtPadAfter,
            self.tr("Automatically add space after any of these symbols."), stretch=(2, 1)
        )

        # Use Thin Space
        self.fmtPadThin = NSwitch(self)
        self.fmtPadThin.setChecked(CONFIG.fmtPadThin)
        self.fmtPadThin.setEnabled(CONFIG.doReplace)
        self.mainForm.addRow(
            self.tr("Use thin space instead"), self.fmtPadThin,
            self.tr("Inserts a thin space instead of a regular space.")
        )

        # Quotation Style
        # ===============

        title = self.tr("Quotation Style")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        # Single Quote Style
        self.fmtSQuoteOpen = QLineEdit(self)
        self.fmtSQuoteOpen.setMaxLength(1)
        self.fmtSQuoteOpen.setReadOnly(True)
        self.fmtSQuoteOpen.setFixedWidth(boxFixed)
        self.fmtSQuoteOpen.setAlignment(QtAlignCenter)
        self.fmtSQuoteOpen.setText(CONFIG.fmtSQuoteOpen)
        self.btnSQuoteOpen = NIconToolButton(self, iSz, "quote")
        self.btnSQuoteOpen.clicked.connect(self._changeSingleQuoteOpen)
        self.mainForm.addRow(
            self.tr("Single quote open style"), self.fmtSQuoteOpen,
            self.tr("The symbol to use for a leading single quote."),
            button=self.btnSQuoteOpen
        )

        self.fmtSQuoteClose = QLineEdit(self)
        self.fmtSQuoteClose.setMaxLength(1)
        self.fmtSQuoteClose.setReadOnly(True)
        self.fmtSQuoteClose.setFixedWidth(boxFixed)
        self.fmtSQuoteClose.setAlignment(QtAlignCenter)
        self.fmtSQuoteClose.setText(CONFIG.fmtSQuoteClose)
        self.btnSQuoteClose = NIconToolButton(self, iSz, "quote")
        self.btnSQuoteClose.clicked.connect(self._changeSingleQuoteClose)
        self.mainForm.addRow(
            self.tr("Single quote close style"), self.fmtSQuoteClose,
            self.tr("The symbol to use for a trailing single quote."),
            button=self.btnSQuoteClose
        )

        # Double Quote Style
        self.fmtDQuoteOpen = QLineEdit(self)
        self.fmtDQuoteOpen.setMaxLength(1)
        self.fmtDQuoteOpen.setReadOnly(True)
        self.fmtDQuoteOpen.setFixedWidth(boxFixed)
        self.fmtDQuoteOpen.setAlignment(QtAlignCenter)
        self.fmtDQuoteOpen.setText(CONFIG.fmtDQuoteOpen)
        self.btnDQuoteOpen = NIconToolButton(self, iSz, "quote")
        self.btnDQuoteOpen.clicked.connect(self._changeDoubleQuoteOpen)
        self.mainForm.addRow(
            self.tr("Double quote open style"), self.fmtDQuoteOpen,
            self.tr("The symbol to use for a leading double quote."),
            button=self.btnDQuoteOpen
        )

        self.fmtDQuoteClose = QLineEdit(self)
        self.fmtDQuoteClose.setMaxLength(1)
        self.fmtDQuoteClose.setReadOnly(True)
        self.fmtDQuoteClose.setFixedWidth(boxFixed)
        self.fmtDQuoteClose.setAlignment(QtAlignCenter)
        self.fmtDQuoteClose.setText(CONFIG.fmtDQuoteClose)
        self.btnDQuoteClose = NIconToolButton(self, iSz, "quote")
        self.btnDQuoteClose.clicked.connect(self._changeDoubleQuoteClose)
        self.mainForm.addRow(
            self.tr("Double quote close style"), self.fmtDQuoteClose,
            self.tr("The symbol to use for a trailing double quote."),
            button=self.btnDQuoteClose
        )

        # AI Preferences
        # ==============

        ai_config = CONFIG.ai
        ai_available = not hasattr(ai_config, '_reason')  # _DisabledAIConfig has _reason attribute

        self.sidebar.addLabel(self.tr("AI"))

        title = self.tr("AI Configuration")
        section += 1
        self.sidebar.addButton(title, section)
        self.mainForm.addGroupLabel(title, section)

        self.aiEnabled = NSwitch(self)
        self.aiEnabled.setObjectName("aiEnabledSwitch")
        self.aiEnabled.setChecked(bool(getattr(ai_config, "enabled", False)))
        self.aiEnabled.toggled.connect(self._update_ai_controls)
        self.mainForm.addRow(
            self.tr("Enable AI Copilot"), self.aiEnabled,
            self.tr("Turn on to expose AI-assisted features in the UI."),
        )

        self.aiDisabledMessage = NColorLabel(
            self.tr("AI Copilot is disabled. Enable the switch above to configure providers."),
            self,
            color=SHARED.theme.errorText,
            wrap=True,
        )
        self.aiDisabledMessage.setObjectName("aiDisabledMessage")
        self.mainForm.addRow(None, self.aiDisabledMessage)

        if not hasattr(self, "_openai_sdk_available"):
            self._openai_sdk_available = self._probe_openai_sdk()

        self.aiProvider = NComboBox(self)
        self.aiProvider.setObjectName("aiProviderCombo")
        self.aiProvider.setMinimumWidth(220)
        provider_options = [
            (self.tr("OpenAI (SDK)"), "openai"),
        ]
        seen_values = {value for _, value in provider_options}
        provider_current = getattr(ai_config, "provider", "openai")
        if provider_current not in seen_values:
            provider_options.append((provider_current.title(), provider_current))
        for label, value in provider_options:
            self.aiProvider.addItem(label, value)
        self.aiProvider.currentIndexChanged.connect(self._on_ai_provider_changed)
        if self.aiProvider.findData(provider_current) >= 0:
            self.aiProvider.setCurrentData(provider_current, "openai")
        else:
            self.aiProvider.setCurrentIndex(0)
        self.mainForm.addRow(
            self.tr("Provider"), self.aiProvider,
            self.tr("Select the AI provider or compatibility mode."), stretch=(3, 2)
        )

        self.aiProviderAvailability = NColorLabel("", self, color=SHARED.theme.helpText, wrap=True)
        self.aiProviderAvailability.setObjectName("aiProviderAvailabilityLabel")
        self.aiProviderAvailability.setVisible(False)
        self.mainForm.addRow(None, self.aiProviderAvailability)
        self._update_provider_availability_message()

        self.aiBaseUrl = QLineEdit(self)
        self.aiBaseUrl.setObjectName("aiBaseUrlEdit")
        self.aiBaseUrl.setMinimumWidth(260)
        self.aiBaseUrl.setPlaceholderText("https://api.openai.com/v1")
        self.aiBaseUrl.setText(getattr(ai_config, "openai_base_url", ""))
        self.mainForm.addRow(
            self.tr("Base URL"), self.aiBaseUrl,
            self.tr("Override the REST endpoint when using compatible services."), stretch=(3, 2)
        )

        self.aiTimeout = NSpinBox(self)
        self.aiTimeout.setObjectName("aiTimeoutSpin")
        self.aiTimeout.setMinimum(5)
        self.aiTimeout.setMaximum(600)
        self.aiTimeout.setSingleStep(5)
        self.aiTimeout.setValue(getattr(ai_config, "timeout", 30))
        self.mainForm.addRow(
            self.tr("Request timeout"), self.aiTimeout,
            self.tr("Abort network calls after the specified number of seconds."), unit=self.tr("seconds")
        )

        self.aiMaxTokens = NSpinBox(self)
        self.aiMaxTokens.setObjectName("aiMaxTokensSpin")
        self.aiMaxTokens.setMinimum(256)
        self.aiMaxTokens.setMaximum(32768)
        self.aiMaxTokens.setSingleStep(256)
        self.aiMaxTokens.setValue(getattr(ai_config, "max_tokens", 2048))
        self.mainForm.addRow(
            self.tr("Maximum tokens"), self.aiMaxTokens,
            self.tr("Upper limit for generated tokens per request."),
        )

        self.aiDryRunDefault = NSwitch(self)
        self.aiDryRunDefault.setObjectName("aiDryRunSwitch")
        self.aiDryRunDefault.setChecked(getattr(ai_config, "dry_run_default", True))
        self.mainForm.addRow(
            self.tr("Dry-run by default"), self.aiDryRunDefault,
            self.tr("Start AI actions in preview mode, requiring manual confirmation."),
        )

        self.aiAskBeforeApply = NSwitch(self)
        self.aiAskBeforeApply.setObjectName("aiAskBeforeApplySwitch")
        self.aiAskBeforeApply.setChecked(getattr(ai_config, "ask_before_apply", True))
        self.mainForm.addRow(
            self.tr("Ask before applying changes"), self.aiAskBeforeApply,
            self.tr("Always prompt before AI changes affect documents."),
        )

        self.aiApiKey = QLineEdit(self)
        self.aiApiKey.setObjectName("aiApiKeyEdit")
        self.aiApiKey.setEchoMode(QLineEdit.EchoMode.Password)
        self.aiApiKey.setMinimumWidth(260)
        self.aiApiKeyEnvOverride = bool(getattr(ai_config, "api_key_from_env", False))
        if self.aiApiKeyEnvOverride:
            self.aiApiKey.setPlaceholderText(
                self.tr("Using OPENAI_API_KEY from the environment; stored key is ignored.")
            )
            self.aiApiKey.setReadOnly(True)
        else:
            self.aiApiKey.setText(getattr(ai_config, "api_key", ""))
        self.mainForm.addRow(
            self.tr("API key"), self.aiApiKey,
            self.tr("The secret token used to authenticate API requests."), stretch=(3, 2)
        )

        self.aiEnvInfo = NColorLabel(
            self.tr(
                "Environment variables take precedence over stored keys."
                " Set OPENAI_API_KEY to avoid saving credentials to disk."
            ),
            self,
            color=SHARED.theme.helpText,
            wrap=True,
        )
        self.aiEnvInfo.setObjectName("aiEnvInfoLabel")
        self.mainForm.addRow(None, self.aiEnvInfo)

        # Test connection and model selection
        self.aiTestButton = QPushButton(self.tr("Test Connection"), self)
        self.aiTestButton.setObjectName("aiTestButton")
        self.aiTestButton.clicked.connect(self._test_ai_connection)
        self.aiTestStatusLabel = NColorLabel("", self, color=SHARED.theme.helpText, wrap=True)
        self.aiTestStatusLabel.setObjectName("aiTestStatusLabel")
        test_layout = QHBoxLayout()
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.setSpacing(8)
        test_layout.addWidget(self.aiTestButton)
        test_layout.addWidget(self.aiTestStatusLabel, 1)
        test_widget = QWidget()
        test_widget.setLayout(test_layout)
        self.mainForm.addRow(
            self.tr("Connection"), test_widget,
            self.tr("Test the API connection and refresh available models.")
        )

        # Model selection
        self.aiModelSelector = NComboBox(self)
        self.aiModelSelector.setObjectName("aiModelSelector")
        self.aiModelSelector.setMinimumWidth(260)
        self.aiModelSelector.currentTextChanged.connect(self._on_model_selection_changed)
        current_model = getattr(ai_config, "model", "")
        if current_model:
            self.aiModelSelector.addItem(current_model, current_model)
            self.aiModelSelector.setCurrentData(current_model, default="")
        self.mainForm.addRow(
            self.tr("Default Model"), self.aiModelSelector,
            self.tr("Select the default model for AI operations."), stretch=(3, 2)
        )

        self._update_ai_controls(self.aiEnabled.isChecked() and ai_available)

        self.mainForm.finalise()
        self.sidebar.setSelected(1)

    ##
    #  Events
    ##

    def closeEvent(self, event: QCloseEvent) -> None:
        """Capture the close event and perform cleanup."""
        logger.debug("Close: GuiPreferences")
        self._saveWindowSize()
        event.accept()
        self.softDelete()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Overload keyPressEvent and only accept escape. The main
        purpose here is to prevent Enter/Return from closing the dialog
        as it is used for the search box.
        """
        if event.matches(QKeySequence.StandardKey.Cancel):
            self.close()
        event.ignore()

    ##
    #  Private Slots
    ##

    @pyqtSlot(int)
    def _sidebarClicked(self, section: int) -> None:
        """Process a user request to switch page."""
        self.mainForm.scrollToSection(section)

    @pyqtSlot()
    def _gotoSearch(self) -> None:
        """Go to the setting indicated by the search text."""
        self.mainForm.scrollToLabel(self.searchText.text().strip())

    @pyqtSlot()
    def _selectGuiFont(self) -> None:
        """Open the QFontDialog and set a font for the font style."""
        font, status = SHARED.getFont(self._guiFont, self.nativeFont.isChecked())
        if status:
            self.guiFont.setText(describeFont(font))
            self.guiFont.setCursorPosition(0)
            self._guiFont = font

    @pyqtSlot()
    def _selectTextFont(self) -> None:
        """Open the QFontDialog and set a font for the font style."""
        font, status = SHARED.getFont(self._textFont, self.nativeFont.isChecked())
        if status:
            self.textFont.setText(describeFont(font))
            self.textFont.setCursorPosition(0)
            self._textFont = font

    @pyqtSlot()
    def _backupFolder(self) -> None:
        """Open a dialog to select the backup folder."""
        if path := QFileDialog.getExistingDirectory(
            self, self.tr("Backup Directory"), str(self.backupPath) or "",
            options=QFileDialog.Option.ShowDirsOnly
        ):
            self.backupPath = path
            self.mainForm.setHelpText("backupPath", self.tr("Path: {0}").format(path))

    @pyqtSlot(bool)
    def _toggledBackupOnClose(self, state: bool) -> None:
        """Toggle switch that depends on the backup on close switch."""
        self.askBeforeBackup.setEnabled(state)

    @pyqtSlot(str)
    def _insertDialogLineSymbol(self, symbol: str) -> None:
        """Insert a symbol in the dialogue line box."""
        current = self.dialogLine.text()
        values = processDialogSymbols(f"{current} {symbol}")
        self.dialogLine.setText(" ".join(values))

    def _update_ai_controls(self, state: bool) -> None:
        """Enable or disable AI configuration inputs based on switch state."""

        ai_config = CONFIG.ai
        ai_available = not hasattr(ai_config, "_reason")
        provider_id = self.aiProvider.currentData()
        provider_available = self._provider_is_available(provider_id)

        base_enabled = bool(state) and ai_available and provider_available

        controls = [
            self.aiBaseUrl,
            self.aiTimeout,
            self.aiMaxTokens,
            self.aiDryRunDefault,
            self.aiAskBeforeApply,
        ]
        for widget in controls:
            widget.setEnabled(base_enabled)

        self.aiProvider.setEnabled(bool(state) and ai_available)
        
        # Enable/disable individual provider items based on availability
        model = self.aiProvider.model()
        if hasattr(model, 'item'):
            for i in range(self.aiProvider.count()):
                item = model.item(i)
                if item:
                    provider_value = self.aiProvider.itemData(i)
                    provider_available = self._provider_is_available(provider_value)
                    item.setEnabled(provider_available)
        self.aiApiKey.setEnabled(base_enabled and not self.aiApiKeyEnvOverride)

        if hasattr(self, "aiTestButton"):
            self.aiTestButton.setEnabled(base_enabled)
        if hasattr(self, "aiModelSelector"):
            self.aiModelSelector.setEnabled(base_enabled)

        if not ai_available:
            self.aiDisabledMessage.setText(
                self.tr("AI features are not available. Please install the AI dependencies.")
            )
            self.aiDisabledMessage.setVisible(True)
            self.aiEnabled.setEnabled(False)
        else:
            self.aiEnabled.setEnabled(True)
            if not state:
                self.aiDisabledMessage.setText(
                    self.tr("AI Copilot is disabled. Enable the switch above to configure providers.")
                )
                self.aiDisabledMessage.setVisible(True)
            else:
                self.aiDisabledMessage.setVisible(False)

        self._update_provider_availability_message()

    def _provider_is_available(self, provider_id: str | None) -> bool:
        provider = (provider_id or "").strip().lower()
        if provider == "openai":
            return bool(getattr(self, "_openai_sdk_available", False))
        return True

    def _probe_openai_sdk(self) -> bool:
        return importlib.util.find_spec("openai") is not None

    def _update_provider_availability_message(self) -> None:
        # Only show availability message when AI is enabled
        if not self.aiEnabled.isChecked():
            self.aiProviderAvailability.clear()
            self.aiProviderAvailability.setVisible(False)
            return
            
        provider_id = self.aiProvider.currentData()
        sdk_available = getattr(self, "_openai_sdk_available", False)
        if provider_id == "openai" and not sdk_available:
            message = self.tr(
                "Install the official OpenAI SDK (pip install novelWriter[ai]) to enable this provider."
            )
            self.aiProviderAvailability.setText(message)
            self.aiProviderAvailability.setVisible(True)
        else:
            self.aiProviderAvailability.clear()
            self.aiProviderAvailability.setVisible(False)

    @pyqtSlot(int)
    def _on_ai_provider_changed(self, _index: int) -> None:
        self._update_provider_availability_message()
        self._update_ai_controls(self.aiEnabled.isChecked())

    @pyqtSlot()
    def _test_ai_connection(self) -> None:
        """Test AI connection and refresh available models."""
        # Check if AI is available
        ai_config = CONFIG.ai
        ai_available = not hasattr(ai_config, '_reason')
        
        if not ai_available:
            self.aiTestStatusLabel.setText(
                self.tr("AI features are not available. Please install the AI dependencies.")
            )
            if hasattr(self.aiTestStatusLabel, 'setColor'):
                self.aiTestStatusLabel.setColor(SHARED.theme.errorText)
            return
            
        if not self.aiEnabled.isChecked():
            return

        self.aiTestButton.setEnabled(False)
        self.aiTestButton.setText(self.tr("Testing..."))
        self.aiTestStatusLabel.setText(self.tr("Connecting..."))
        if hasattr(self.aiTestStatusLabel, 'setColor'):
            self.aiTestStatusLabel.setColor(SHARED.theme.helpText)

        try:
            # Create temporary config with current form values
            try:
                from novelwriter.ai.config import AIConfig
                from novelwriter.ai import NWAiApi
            except ImportError as exc:
                raise Exception(f"AI modules not available: {exc}")
                
            temp_config = AIConfig()
            temp_config.enabled = True
            temp_config.provider = self.aiProvider.currentData() or "openai"
            temp_config.openai_base_url = self.aiBaseUrl.text().strip()
            temp_config.api_key = self.aiApiKey.text().strip()
            temp_config.timeout = self.aiTimeout.value()

            # Test the connection by listing models
            if SHARED.project:
                api = NWAiApi(SHARED.project)
            else:
                raise Exception("No project available for AI testing")
            
            # Override config temporarily
            original_config = getattr(CONFIG, "ai", None)
            setattr(CONFIG, '_ai_config', temp_config)
            
            try:
                models = api.listAvailableModels(refresh=True)
                self._populate_model_selector(models)
                self.aiTestStatusLabel.setText(
                    self.tr("Connected successfully. Found {0} model(s).").format(len(models))
                )
                if hasattr(self.aiTestStatusLabel, 'setColor') and hasattr(SHARED.theme, 'textGreen'):
                    self.aiTestStatusLabel.setColor(SHARED.theme.textGreen)
            finally:
                # Restore original config
                if original_config is not None:
                    setattr(CONFIG, '_ai_config', original_config)

        except Exception as exc:
            logger.error("AI connection test failed: %s", exc)
            self.aiTestStatusLabel.setText(
                self.tr("Connection failed: {0}").format(str(exc))
            )
            if hasattr(self.aiTestStatusLabel, 'setColor'):
                self.aiTestStatusLabel.setColor(SHARED.theme.errorText)
        finally:
            self.aiTestButton.setEnabled(True)
            self.aiTestButton.setText(self.tr("Test Connection"))

    def _populate_model_selector(self, models: list) -> None:
        """Populate the model selector with available models."""
        if not hasattr(self, 'aiModelSelector'):
            return

        current_selection = self.aiModelSelector.currentData()
        self.aiModelSelector.clear()

        if not models:
            self.aiModelSelector.addItem(self.tr("No models available"), "")
            return

        for model in models:
            if hasattr(model, 'display_name') and hasattr(model, 'id'):
                display_name = model.display_name
                model_id = model.id
                self.aiModelSelector.addItem(display_name, model_id)

        # Restore selection if possible
        if current_selection:
            index = self.aiModelSelector.findData(current_selection)
            if index >= 0:
                self.aiModelSelector.setCurrentIndex(index)

    @pyqtSlot(str)
    def _on_model_selection_changed(self, model_name: str) -> None:
        """Handle model selection change."""
        model_id = self.aiModelSelector.currentData()
        if model_id:
            logger.debug("Model selection changed to: %s (%s)", model_name, model_id)

    @pyqtSlot(bool)
    def _toggleAutoReplaceMain(self, state: bool) -> None:
        """Toggle switches controlled by the auto replace switch."""
        self.doReplaceSQuote.setEnabled(state)
        self.doReplaceDQuote.setEnabled(state)
        self.doReplaceDash.setEnabled(state)
        self.doReplaceDots.setEnabled(state)
        self.fmtPadThin.setEnabled(state)

    @pyqtSlot()
    def _changeSingleQuoteOpen(self) -> None:
        """Change single quote open style."""
        quote, status = GuiQuoteSelect.getQuote(self, current=self.fmtSQuoteOpen.text())
        if status:
            self.fmtSQuoteOpen.setText(quote)

    @pyqtSlot()
    def _changeSingleQuoteClose(self) -> None:
        """Change single quote close style."""
        quote, status = GuiQuoteSelect.getQuote(self, current=self.fmtSQuoteClose.text())
        if status:
            self.fmtSQuoteClose.setText(quote)

    @pyqtSlot()
    def _changeDoubleQuoteOpen(self) -> None:
        """Change double quote open style."""
        quote, status = GuiQuoteSelect.getQuote(self, current=self.fmtDQuoteOpen.text())
        if status:
            self.fmtDQuoteOpen.setText(quote)

    @pyqtSlot()
    def _changeDoubleQuoteClose(self) -> None:
        """Change double quote close style."""
        quote, status = GuiQuoteSelect.getQuote(self, current=self.fmtDQuoteClose.text())
        if status:
            self.fmtDQuoteClose.setText(quote)

    ##
    #  Internal Functions
    ##

    def _saveWindowSize(self) -> None:
        """Save the dialog window size."""
        CONFIG.setPreferencesWinSize(self.width(), self.height())

    def _doSave(self) -> None:
        """Save the values set in the form."""
        updateTheme  = False
        needsRestart = False
        updateSyntax = False
        refreshTree  = False

        # Appearance
        guiLocale    = self.guiLocale.currentData()
        lightTheme   = self.lightTheme.currentData()
        darkTheme    = self.darkTheme.currentData()
        iconTheme    = self.iconTheme.currentData()
        useCharCount = self.useCharCount.isChecked()

        updateTheme  |= CONFIG.lightTheme != lightTheme
        updateTheme  |= CONFIG.darkTheme != darkTheme
        updateTheme  |= CONFIG.iconTheme != iconTheme
        needsRestart |= CONFIG.guiLocale != guiLocale
        needsRestart |= CONFIG.guiFont != self._guiFont
        refreshTree  |= CONFIG.useCharCount != useCharCount
        updateSyntax |= CONFIG.lightTheme != lightTheme
        updateSyntax |= CONFIG.darkTheme != darkTheme

        CONFIG.guiLocale    = guiLocale
        CONFIG.lightTheme   = lightTheme
        CONFIG.darkTheme    = darkTheme
        CONFIG.iconTheme    = iconTheme
        CONFIG.hideVScroll  = self.hideVScroll.isChecked()
        CONFIG.hideHScroll  = self.hideHScroll.isChecked()
        CONFIG.nativeFont   = self.nativeFont.isChecked()
        CONFIG.useCharCount = useCharCount
        CONFIG.setGuiFont(self._guiFont)

        # Document Style
        CONFIG.showFullPath   = self.showFullPath.isChecked()
        CONFIG.incNotesWCount = self.incNotesWCount.isChecked()
        CONFIG.setTextFont(self._textFont)

        # Project View
        iconColTree = self.iconColTree.currentData()
        iconColDocs = self.iconColDocs.isChecked()
        emphLabels = self.emphLabels.isChecked()

        updateTheme |= CONFIG.iconColTree != iconColTree
        updateTheme |= CONFIG.iconColDocs != iconColDocs
        refreshTree |= CONFIG.emphLabels != emphLabels

        CONFIG.iconColTree = iconColTree
        CONFIG.iconColDocs = iconColDocs
        CONFIG.emphLabels     = emphLabels

        # Behaviour
        CONFIG.autoSaveDoc   = self.autoSaveDoc.value()
        CONFIG.autoSaveProj  = self.autoSaveProj.value()
        CONFIG.askBeforeExit = self.askBeforeExit.isChecked()

        # Project Backup
        CONFIG.setBackupPath(self.backupPath)
        CONFIG.backupOnClose   = self.backupOnClose.isChecked()
        CONFIG.askBeforeBackup = self.askBeforeBackup.isChecked()

        # Session Timer
        CONFIG.stopWhenIdle = self.stopWhenIdle.isChecked()
        CONFIG.userIdleTime = round(self.userIdleTime.value() * 60)

        # Text Flow
        CONFIG.textWidth       = self.textWidth.value()
        CONFIG.focusWidth      = self.focusWidth.value()
        CONFIG.hideFocusFooter = self.hideFocusFooter.isChecked()
        CONFIG.doJustify       = self.doJustify.isChecked()
        CONFIG.textMargin      = self.textMargin.value()
        CONFIG.tabWidth        = self.tabWidth.value()

        # Text Editing
        lineHighlight = self.lineHighlight.isChecked()

        updateSyntax |= CONFIG.lineHighlight != lineHighlight

        CONFIG.spellLanguage   = self.spellLanguage.currentData()
        CONFIG.autoSelect      = self.autoSelect.isChecked()
        CONFIG.cursorWidth     = self.cursorWidth.value()
        CONFIG.lineHighlight   = lineHighlight
        CONFIG.showTabsNSpaces = self.showTabsNSpaces.isChecked()
        CONFIG.showLineEndings = self.showLineEndings.isChecked()

        # Editor Scrolling
        CONFIG.autoScroll    = self.autoScroll.isChecked()
        CONFIG.autoScrollPos = self.autoScrollPos.value()
        CONFIG.scrollPastEnd = self.scrollPastEnd.isChecked()

        # Text Highlighting
        dialogueStyle   = self.dialogStyle.currentData()
        allowOpenDial   = self.allowOpenDial.isChecked()
        dialogueLine    = processDialogSymbols(self.dialogLine.text())
        narratorBreak   = self.narratorBreak.currentData()
        narratorDialog  = self.narratorDialog.currentData()
        altDialogOpen   = compact(self.altDialogOpen.text())
        altDialogClose  = compact(self.altDialogClose.text())
        highlightEmph   = self.highlightEmph.isChecked()
        showMultiSpaces = self.showMultiSpaces.isChecked()

        updateSyntax |= CONFIG.dialogStyle != dialogueStyle
        updateSyntax |= CONFIG.allowOpenDial != allowOpenDial
        updateSyntax |= CONFIG.dialogLine != dialogueLine
        updateSyntax |= CONFIG.narratorBreak != narratorBreak
        updateSyntax |= CONFIG.narratorDialog != narratorDialog
        updateSyntax |= CONFIG.altDialogOpen != altDialogOpen
        updateSyntax |= CONFIG.altDialogClose != altDialogClose
        updateSyntax |= CONFIG.highlightEmph != highlightEmph
        updateSyntax |= CONFIG.showMultiSpaces != showMultiSpaces

        CONFIG.dialogStyle     = dialogueStyle
        CONFIG.allowOpenDial   = allowOpenDial
        CONFIG.dialogLine      = dialogueLine
        CONFIG.narratorBreak   = narratorBreak
        CONFIG.narratorDialog  = narratorDialog
        CONFIG.altDialogOpen   = altDialogOpen
        CONFIG.altDialogClose  = altDialogClose
        CONFIG.highlightEmph   = highlightEmph
        CONFIG.showMultiSpaces = showMultiSpaces

        # Text Automation
        CONFIG.doReplace       = self.doReplace.isChecked()
        CONFIG.doReplaceSQuote = self.doReplaceSQuote.isChecked()
        CONFIG.doReplaceDQuote = self.doReplaceDQuote.isChecked()
        CONFIG.doReplaceDash   = self.doReplaceDash.isChecked()
        CONFIG.doReplaceDots   = self.doReplaceDots.isChecked()
        CONFIG.fmtPadBefore    = uniqueCompact(self.fmtPadBefore.text())
        CONFIG.fmtPadAfter     = uniqueCompact(self.fmtPadAfter.text())
        CONFIG.fmtPadThin      = self.fmtPadThin.isChecked()

        # Quotation Style
        CONFIG.fmtSQuoteOpen  = self.fmtSQuoteOpen.text()
        CONFIG.fmtSQuoteClose = self.fmtSQuoteClose.text()
        CONFIG.fmtDQuoteOpen  = self.fmtDQuoteOpen.text()
        CONFIG.fmtDQuoteClose = self.fmtDQuoteClose.text()

        # AI Configuration
        try:
            ai_config = CONFIG.ai
            ai_available = not hasattr(ai_config, '_reason')
            
            if ai_available and hasattr(self, 'aiEnabled'):
                ai_config.enabled = self.aiEnabled.isChecked()
                if hasattr(self, 'aiProvider'):
                    ai_config.provider = self.aiProvider.currentData() or ai_config.provider
                if hasattr(self, 'aiBaseUrl'):
                    base_url = self.aiBaseUrl.text().strip()
                    ai_config.openai_base_url = base_url or "https://api.openai.com/v1"
                if hasattr(self, 'aiTimeout'):
                    ai_config.timeout = self.aiTimeout.value()
                if hasattr(self, 'aiMaxTokens'):
                    ai_config.max_tokens = self.aiMaxTokens.value()
                if hasattr(self, 'aiDryRunDefault'):
                    ai_config.dry_run_default = self.aiDryRunDefault.isChecked()
                if hasattr(self, 'aiAskBeforeApply'):
                    ai_config.ask_before_apply = self.aiAskBeforeApply.isChecked()
                if hasattr(self, 'aiApiKeyEnvOverride') and hasattr(self, 'aiApiKey'):
                    if not self.aiApiKeyEnvOverride:
                        ai_config.api_key = self.aiApiKey.text().strip()
                
                # Save selected model
                if hasattr(self, 'aiModelSelector'):
                    selected_model_id = self.aiModelSelector.currentData()
                    if selected_model_id and isinstance(selected_model_id, str):
                        ai_config.model = selected_model_id
        except Exception as exc:
            # In test environments or when AI modules are unavailable,
            # we should not let AI configuration errors affect other settings
            logger.debug("Failed to save AI configuration: %s", exc)

        # Finalise
        CONFIG.saveConfig()

        try:
            from novelwriter.extensions.ai_copilot.dock import AICopilotDock
        except Exception:  # pragma: no cover - optional integration
            pass
        else:
            dock = SHARED.findTopLevelWidget(AICopilotDock)
            if dock is not None:
                dock.refresh_from_config()

        self.newPreferencesReady.emit(needsRestart, refreshTree, updateTheme, updateSyntax)

        self.close()
