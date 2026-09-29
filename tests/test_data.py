import pytest

from pxai.data import (
    audit_random_split,
    index_kermany,
    make_splits,
    parse_kermany_filename,
    patient_overlap,
)


@pytest.mark.parametrize(
    ("name", "label", "expected"),
    [
        ("person1234_bacteria_5678.jpeg", "PNEUMONIA", ("PNEUMONIA:person1234", "bacteria")),
        ("person7_virus_30.jpeg", "PNEUMONIA", ("PNEUMONIA:person7", "virus")),
        ("IM-0115-0001.jpeg", "NORMAL", ("NORMAL:IM-0115", None)),
        ("NORMAL2-IM-1427-0001.jpeg", "NORMAL", ("NORMAL:NORMAL2-IM-1427", None)),
        ("weird.png", "NORMAL", ("NORMAL:weird", None)),
    ],
)
def test_parse_kermany_filename(name, label, expected):
    assert parse_kermany_filename(name, label) == expected


def test_index_finds_all_images(kermany_tree):
    df = index_kermany(kermany_tree)
    assert len(df) == 24 + 40 + 5 + 7
    assert set(df.official_split) == {"train", "test"}
    assert df.groupby("patient_id").size().max() >= 2


@pytest.mark.parametrize("mode", ["official", "patient"])
def test_splits_have_no_patient_overlap(kermany_tree, mode):
    df = make_splits(index_kermany(kermany_tree), mode=mode, val_fraction=0.2, test_fraction=0.2)
    assert set(df.split) == {"train", "val", "test"}
    assert patient_overlap(df) == {"train&val": 0, "train&test": 0, "val&test": 0}
    # both classes present in every split
    assert (df.groupby("split").label.nunique() == 2).all()


def test_official_mode_drops_patient_shared_with_test(kermany_tree):
    df = make_splits(index_kermany(kermany_tree), mode="official")
    person1 = df[df.patient_id == "PNEUMONIA:person1"]
    assert set(person1.split) == {"test"}


def test_splits_are_reproducible(kermany_tree):
    a = make_splits(index_kermany(kermany_tree), mode="patient", seed=3)
    b = make_splits(index_kermany(kermany_tree), mode="patient", seed=3)
    assert (a.split == b.split).all()


def test_random_image_split_leaks(kermany_tree):
    report = audit_random_split(index_kermany(kermany_tree), test_fraction=0.3)
    assert report["leak_fraction"] > 0
