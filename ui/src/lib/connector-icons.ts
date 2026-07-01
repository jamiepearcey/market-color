import { Cable, Cloud, Database, FileText, LineChart, Radio, ScrollText } from "lucide-react";

/**
 * Lucide icon for a connector category — shared by the connector catalog, the
 * job connector picker, and the packs browser so they stay visually identical.
 * Lives outside any component module so React Fast Refresh stays happy.
 */
export function iconForCategory(category: string) {
  switch (category) {
    case "Filesystem":
      return FileText;
    case "Object Store":
      return Cloud;
    case "Databases":
      return Database;
    case "Streaming":
      return Radio;
    case "Consolidated Tape":
      return ScrollText;
    case "Vendor":
      return LineChart;
    default:
      return Cable;
  }
}
