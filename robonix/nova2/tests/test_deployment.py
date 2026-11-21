"""Cross-check the local deployment, body description and package capabilities."""

import shlex
import unittest
from pathlib import Path
from xml.etree import ElementTree

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def walk_parts(parts):
    for part in parts:
        yield part
        yield from walk_parts(part.get("components", []))


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.deployment = read_yaml(ROOT / "robonix_manifest.yaml")
        body_path = ROOT / self.deployment["system"]["soma"]["robot_yaml"]
        self.body = read_yaml(body_path)
        self.urdf = ElementTree.parse(body_path.parent / self.body["urdf"]["path"])

    def test_native_entry_scripts_have_unix_line_endings(self):
        for package_path in {entry["path"] for entry in self.deployment["primitive"]}:
            package = ROOT / package_path
            manifest = read_yaml(package / "package_manifest.yaml")
            for action in ("build", "start"):
                program, script = shlex.split(manifest[action])
                self.assertEqual(program, "bash")
                with self.subTest(script=str(package / script)):
                    data = (package / script).read_bytes()
                    self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
                    self.assertNotIn(b"\r", data)

    def test_soma_exports_match_every_package_instance(self):
        entries = self.deployment["primitive"]
        expected = {f"{side}_{kind}" for side in ("left", "right") for kind in ("arm", "gripper")}
        self.assertCountEqual([entry["name"] for entry in entries], expected)
        exports = {}
        for part in walk_parts(self.body["robot"]["components"]):
            for export in part.get("exports", []):
                names = exports.setdefault(export["provider_id"], set())
                names.update(cap["path"] for cap in export["capabilities"])
        self.assertEqual(set(exports), expected)
        for entry in entries:
            name = entry["name"]
            with self.subTest(provider=name):
                side, kind = name.split("_")
                self.assertEqual(entry["config"]["side"], side)
                package = read_yaml(ROOT / entry["path"] / "package_manifest.yaml")
                contracts = {cap["name"] for cap in package["capabilities"]}
                self.assertEqual(contracts, exports[name])
                operations = ("joint_states", "joint_command") if kind == "arm" else (
                    "info", "open", "move", "grip", "stop", "state"
                )
                self.assertEqual(contracts, {f"robonix/primitive/{kind}/{op}" for op in operations})

    def test_urdf_is_one_tree_with_twelve_named_arm_joints(self):
        links = [node.attrib["name"] for node in self.urdf.findall("link")]
        joints = self.urdf.findall("joint")
        joint_names = [joint.attrib["name"] for joint in joints]
        self.assertEqual(len(set(links)), len(links))
        self.assertEqual(len(set(joint_names)), len(joint_names))
        parents = {}
        for joint in joints:
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            self.assertIn(parent, links)
            self.assertIn(child, links)
            self.assertNotIn(child, parents)
            parents[child] = parent
        root = self.body["urdf"]["root_link"]
        self.assertEqual(set(links) - set(parents), {root})
        for link in links:
            chain = set()
            while link != root:
                self.assertNotIn(link, chain, "URDF contains a cycle")
                chain.add(link)
                link = parents[link]
        for side in ("left", "right"):
            for index in range(1, 7):
                joint = self.urdf.find(f"joint[@name='{side}_joint{index}']")
                self.assertIsNotNone(joint)
                self.assertEqual(joint.attrib["type"], "revolute")
        for part in walk_parts(self.body["robot"]["components"]):
            if "urdf_link" in part:
                self.assertIn(part["urdf_link"], links)
            if "urdf_joint" in part:
                self.assertIn(part["urdf_joint"], joint_names)
