from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from promoapp.models import PromoOutput, VideoUpload

INPUT_RETENTION = timedelta(hours=24)
OUTPUT_RETENTION = timedelta(days=7)


class Command(BaseCommand):
    """Deletes expired media files per DATABASE_SCHEMA.md §5. DB rows are kept indefinitely —
    they're small and they're the portfolio evidence (scores, params, eval results)."""

    help = "Delete input files older than 24h and output files older than 7 days; keep all DB rows."

    def handle(self, *args, **options):
        input_deleted = self._sweep(VideoUpload.objects.filter(created_at__lt=timezone.now() - INPUT_RETENTION))
        output_deleted = self._sweep(PromoOutput.objects.filter(created_at__lt=timezone.now() - OUTPUT_RETENTION))
        self.stdout.write(
            f"Deleted {input_deleted} input file(s) older than 24h, "
            f"{output_deleted} output file(s) older than 7 days. DB rows preserved."
        )

    def _sweep(self, queryset) -> int:
        deleted = 0
        for obj in queryset:
            if not obj.file:
                continue
            try:
                obj.file.delete(save=True)
                deleted += 1
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.stderr.write(f"couldn't delete {obj.file.name}: {exc}")
        return deleted
