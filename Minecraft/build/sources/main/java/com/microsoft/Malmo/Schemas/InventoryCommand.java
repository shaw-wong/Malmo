//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2018.03.20 at 04:41:00 PM AEDT 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for InventoryCommand.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="InventoryCommand">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="swapInventoryItems"/>
 *     &lt;enumeration value="combineInventoryItems"/>
 *     &lt;enumeration value="discardCurrentItem"/>
 *     &lt;enumeration value="hotbar.1"/>
 *     &lt;enumeration value="hotbar.2"/>
 *     &lt;enumeration value="hotbar.3"/>
 *     &lt;enumeration value="hotbar.4"/>
 *     &lt;enumeration value="hotbar.5"/>
 *     &lt;enumeration value="hotbar.6"/>
 *     &lt;enumeration value="hotbar.7"/>
 *     &lt;enumeration value="hotbar.8"/>
 *     &lt;enumeration value="hotbar.9"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "InventoryCommand")
@XmlEnum
public enum InventoryCommand {

    @XmlEnumValue("swapInventoryItems")
    SWAP_INVENTORY_ITEMS("swapInventoryItems"),
    @XmlEnumValue("combineInventoryItems")
    COMBINE_INVENTORY_ITEMS("combineInventoryItems"),
    @XmlEnumValue("discardCurrentItem")
    DISCARD_CURRENT_ITEM("discardCurrentItem"),
    @XmlEnumValue("hotbar.1")
    HOTBAR_1("hotbar.1"),
    @XmlEnumValue("hotbar.2")
    HOTBAR_2("hotbar.2"),
    @XmlEnumValue("hotbar.3")
    HOTBAR_3("hotbar.3"),
    @XmlEnumValue("hotbar.4")
    HOTBAR_4("hotbar.4"),
    @XmlEnumValue("hotbar.5")
    HOTBAR_5("hotbar.5"),
    @XmlEnumValue("hotbar.6")
    HOTBAR_6("hotbar.6"),
    @XmlEnumValue("hotbar.7")
    HOTBAR_7("hotbar.7"),
    @XmlEnumValue("hotbar.8")
    HOTBAR_8("hotbar.8"),
    @XmlEnumValue("hotbar.9")
    HOTBAR_9("hotbar.9");
    private final String value;

    InventoryCommand(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static InventoryCommand fromValue(String v) {
        for (InventoryCommand c: InventoryCommand.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
