"""La tarea y los 20 tickets de prueba."""
import unittest

from tarea import CATEGORIES, LEAK_HINTS, PRIORITIES, QUESTIONS, TICKETS, light, ticket_state


class TaskChecks(unittest.TestCase):
    def test_twenty_tickets_with_valid_references(self):
        self.assertEqual(len({t["id"] for t in TICKETS}), 20)
        for t in TICKETS:
            category, priority, expert = t["referencia"]
            self.assertIn(priority, PRIORITIES)
            if category:
                self.assertEqual(CATEGORIES[category][1], expert, t["id"])  # El experto es el responsable de la categoría.
            self.assertIsInstance(t["bloquea"], bool)
            self.assertNotIn(str(t["referencia"]), str(ticket_state(t)))

    def test_descriptions_do_not_leak_the_answer(self):
        # Un conjunto de prueba que dice la respuesta en el texto no mide nada.
        for t in TICKETS:
            text = t["descripcion"].lower()
            self.assertFalse([h for h in LEAK_HINTS if h in text], t["id"])
            if t["referencia"][0]:
                self.assertNotIn(CATEGORIES[t["referencia"][0]][0].lower(), text, t["id"])

    def test_three_question_types(self):
        self.assertEqual([q["type"] for q in QUESTIONS.values()], ["choice", "score", "noul"])

    def test_light_thresholds(self):
        self.assertEqual([light(c)[0] for c in (100, 80.1, 80, 60, 59.9)], ["verde", "verde", "amarillo", "amarillo", "rojo"])


if __name__ == "__main__":
    unittest.main()
