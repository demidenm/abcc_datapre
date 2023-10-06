from pathlib import Path
import pandas as pd
import datetime
import os.path as op
from pkg_resources import resource_filename as pkgrf
from io import StringIO as TextIO
from collections import OrderedDict

BIDS_COMP = OrderedDict(
        [
            ("subject_id", 'sub'),
            ("session_id", 'ses'),
            ("task_id", 'task'),
        ]
    )

class GroupTemplate:
    """Utilities: Jinja2 templates."""
    """Specific template for the individual report"""

    """
    Utility class for generating a config file from a jinja template.
    https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1f11fd0bd84342d604/endofday/core/template.py
    """

    def __init__(self):
        import jinja2

        self.template_str = pkgrf("scripts", "templates/group.html")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath="/"),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def compile(self, configs):
        """Generates a string with the replacements"""
        template = self.env.get_template(self.template_str)
        return template.render(configs)

    def generate_conf(self, configs, path):
        """Saves the oucome after replacement on the template to file"""
        Path(path).write_text(self.compile(configs))

def gen_html(csv_file, mod, n_subjects, runs, description, qc_items, out_file):

    def_comps = list(BIDS_COMP.keys())
    dataframe = pd.read_csv(
        csv_file, index_col=False, dtype={comp: object for comp in def_comps}
    )
    id_labels = list(set(def_comps) & set(dataframe.columns.to_numpy()))
    dataframe["label"] = dataframe[id_labels].apply(
        _format_labels, args=(id_labels,), axis=1
    )

    nPart = len(dataframe)

    csv_groups = []
    datacols = dataframe.columns.to_numpy()
    for group, units in qc_items[mod]:
        dfdict = {"iqm": [], "value": [], "label": [], "units": [], "image_path": []}

        for iqm in group:
            if iqm in datacols:
                values = dataframe[[iqm]].values.ravel().tolist()
                if values:
                    dfdict["iqm"] += [iqm] * nPart
                    dfdict["units"] += [units] * nPart
                    dfdict["value"] += values
                    dfdict["label"] += dataframe[["label"]].values.ravel().tolist()
                    dfdict["image_path"] += dataframe[["image_path"]].values.ravel().tolist()

        # Save only if there are values
        if dfdict["value"]:
            csv_df = pd.DataFrame(dfdict)
            csv_str = TextIO()
            csv_df[["iqm", "value", "label", "units","image_path"]].to_csv(csv_str, index=False)
            csv_groups.append(csv_str.getvalue())

    if out_file is None:
        out_file = op.abspath("group.html")
    tpl = GroupTemplate()
    tpl.generate_conf(
        {
            "modality": mod,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "subjects": n_subjects,
            "runs": runs,
            "summary_task": description,
            "csv_groups": csv_groups,
            "boxplots_js": open(
                pkgrf("scripts", "embed_resources/boxplots.js"),

            ).read(),
            "d3_js": open(
                pkgrf("scripts", "embed_resources/d3.min.js"),
            ).read(),
            "boxplots_css": open(
                pkgrf("scripts", "embed_resources/boxplots.css")
            ).read(),
        },
        out_file,
    )

    return out_file

def _format_labels(row, id_labels):
    """format participant labels"""
    crow = []

    for col_id, prefix in list(BIDS_COMP.items()):
        if col_id in id_labels:
            crow.append(f"{prefix}-{row[[col_id]].values[0]}")
    return "_".join(crow)




