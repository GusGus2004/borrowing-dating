from pipeline.pmed import norm, sense_keys, senses


def test_norm_keeps_enye_and_strips_accent_marks():
    assert norm("un*a") == "uña"          # PMED writes ñ as n*
    assert norm("pestan*a") == "pestaña"
    assert norm("a*rbol") == "arbol"
    assert norm("Corazón") == "corazon"


def test_senses_split_and_clean():
    assert senses("hierro, piedra//") == ["hierro", "piedra"]
    assert senses("//to learn") == ["learn"]
    assert senses("diente (molar); el ojo//eye") == ["diente", "ojo", "eye"]
    assert "diente" not in senses("chile diente de perro//kind of chile")


def test_enclitics_reduce_to_infinitive():
    assert sense_keys("morderlo") == ["morderlo", "morder"]
    assert sense_keys("dormirse") == ["dormirse", "dormir"]
    assert sense_keys("sangre") == ["sangre"]
