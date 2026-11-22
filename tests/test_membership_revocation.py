"""Consequential store operations re-check current membership authority."""
import tempfile
import unittest
from pathlib import Path

from moneyguard.domain import ValidationError
from moneyguard.selftest import SHAPLA, TARGET, _mandate, _ops
from moneyguard.service import MoneyGuard


class MembershipRevocationTests(unittest.TestCase):
    def test_no_demo_does_not_advertise_fixture_logins(self):
        from moneyguard.service import MoneyGuard
        for seed in (True, False):
            with self.subTest(seed=seed):
                app = MoneyGuard(":memory:", seed=seed)
                try:
                    accounts = app.config()["demo_accounts"]
                    self.assertEqual({a["id"] for a in accounts}, {"nadia", "imran", "guest"} if seed else set())
                finally:
                    app.store.close()

    def test_removal_survives_restart_and_clears_pending_vote(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.sqlite3"
            app = MoneyGuard(path)
            owner, accountant, _ = _ops(app)
            plan = app.propose(SHAPLA, TARGET, "agent:revocation")
            app.store.approve(SHAPLA, plan["plan_id"], "recommended", accountant, "1357", "operator:imran")
            app.store.remove_member(SHAPLA, owner, accountant["id"])
            app.store.close()
            app = MoneyGuard(path)
            self.assertIsNone(app.store.membership_role(accountant["id"], SHAPLA))
            self.assertEqual(app.store.membership_role(accountant["id"], "lotus-print"), "accountant")
            result = app.store.approve(SHAPLA, plan["plan_id"], "recommended", owner, "2468", "operator:nadia")
            self.assertEqual(result["status"], "awaiting_second_approval")
            self.assertEqual([vote["operator_id"] for vote in result["approvals"]], [owner["id"]])
            app.store.close()

    def test_issuance_rechecks_existing_voter_role(self):
        app = MoneyGuard(":memory:")
        owner, accountant, _ = _ops(app)
        plan = app.propose(SHAPLA, TARGET, "agent:revocation")
        app.store.approve(SHAPLA, plan["plan_id"], "recommended", accountant, "1357", "operator:imran")
        with app.store.connect() as db:
            db.execute("UPDATE memberships SET role='viewer' WHERE business_id=? AND operator_id=?",
                       (SHAPLA, accountant["id"]))
        result = app.store.approve(SHAPLA, plan["plan_id"], "recommended", owner, "2468", "operator:nadia")
        self.assertEqual(result["status"], "awaiting_second_approval")
        self.assertEqual([vote["operator_id"] for vote in result["approvals"]], [owner["id"]])

    def test_removed_approver_cannot_leave_a_stale_approval(self):
        app = MoneyGuard(":memory:")
        owner, accountant, _ = _ops(app)
        plan = app.propose(SHAPLA, TARGET, "agent:revocation")

        app.store.remove_member(SHAPLA, owner, accountant["id"])

        with self.assertRaisesRegex(ValidationError, "active membership"):
            app.store.approve(SHAPLA, plan["plan_id"], "recommended", accountant, "1357", "operator:imran")

        with app.store.connect() as db:
            self.assertIsNone(db.execute("SELECT 1 FROM approvals WHERE plan_id=?", (plan["plan_id"],)).fetchone())

    def test_removed_approver_cannot_execute_or_cancel_a_live_mandate(self):
        app = MoneyGuard(":memory:")
        owner, accountant, _ = _ops(app)
        _, mandate = _mandate(app, SHAPLA)
        app.negotiate(SHAPLA, mandate["mandate_id"], "agent:revocation")
        app.store.remove_member(SHAPLA, owner, accountant["id"])

        self.assertEqual(app.store.mandate(SHAPLA, mandate["mandate_id"])["status"], "revoked")
        with self.assertRaisesRegex(ValidationError, "active membership"):
            app.store.execute(SHAPLA, mandate["mandate_id"], "revoked-execute", accountant, "operator:imran")
        with self.assertRaisesRegex(ValidationError, "active membership"):
            app.store.cancel_mandate(SHAPLA, mandate["mandate_id"], accountant, "operator:imran")


if __name__ == "__main__":
    unittest.main()
